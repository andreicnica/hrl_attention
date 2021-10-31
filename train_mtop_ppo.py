import torch
import sys
import gym

from argparse import Namespace
import numpy as np
import glob
import re
import os
import copy

try:
    import gym_minigrid
except ImportError:
    pass

from models import get_model
from agents import get_agent
from utils import utils, get_wrappers, save_training, get_obss_pre_processor
from utils.utils import add_to_cfg
from utils.utils import parse_opts

from utils.logging_utils import LoggerMaster
from utils.utils import flatten_cfg

from train_main import run as train_run

from utils.gym_wrappers import IntentsChangeRoom

MAIN_CFG_ARGS = ["main", "env_cfg", "agent", "model"]

import copy
from copy import deepcopy as cp

EPS = 0.000003


def get_door_pos(obs, dir, room):
    room_size = 6
    center_room = room * room_size + room_size // 2
    op_act = IntentsChangeRoom._change_room[dir].numpy()
    door_pos = center_room + op_act * (room_size // 2)
    elem = obs["image"][door_pos[0], door_pos[1]]
    if elem[0] == 1:
        return True
    return False


class OptionsAgent:
    def __init__(self, opts: Namespace, model_id: int, local_cfg: Namespace,
                 multitask_frame_steps=0, gt_affordable=False, oracle_actions=False):
        # Load necessary config & model
        algo, model, envs, saver = train_run(opts, return_models=True, load_model_id=model_id)
        self.intent_max_horizon = opts.agent.intent_max_horizon
        self.terminate_th = opts.agent.terminate_th
        self.r_discount = getattr(opts.agent, "option_discount", 1)
        self.terminate_op_gt = getattr(opts.agent, "terminate_op_gt", False)

        self.gt_affordable = gt_affordable
        self.device = algo.device
        self.algo = algo
        self.saver = saver
        self.modulate_policy = local_cfg.main.modulate_policy
        self.affordance_th = local_cfg.main.affordance_th
        self.terminate_any = getattr(local_cfg.main, "terminate_any", False)
        self.multitask_frame_steps = multitask_frame_steps
        self.multitask_last_frame = 0

        env = envs[0][0]
        self.num_procs = algo.num_procs
        self.first_env = env

        self._crt_step = 0
        self.all_r = 0

        total_env_cnt = algo.env.no_envs + 1
        self.env_idxs = np.array(range(total_env_cnt))  # Total number of envs
        self.running_envs = torch.zeros(total_env_cnt).bool().to(algo.device)
        self.op_steps = torch.zeros(total_env_cnt).to(algo.device)

        self.mem = [list() for _ in range(self.num_procs)]
        self._last_env_step_cnt = 0

        self._total_frames = 0
        self.step_obs_stack = list()
        self.step_op_stack = [list() for _ in range(self.num_procs)]
        self.step_r_stack = list()
        self.step_done_stack = list()

        self._simulation_inner_step = 0
        self._simulation_outer_step = 0

        self._cnt_tr = 0
        self._cnt_bad_tr = 0
        self._cnt_act_bad_tr = 0

        self._oracle_actions = oracle_actions

        # We run options guided by Policy over options
        algo.model.use_affordances = False
        algo.model.sample_attention_op = False

        if oracle_actions:
            algo.model.options.oracle_actions = True
        # # TODO DEBUG - model affordable for multiobj env
        # self.affordable = dict({"aff": [], "non_aff": []})

    def step(self, op_ids):
        raise NotImplementedError

    def standard_step(self, op_ids):
        algo = self.algo
        device = self.device
        intent_max_horizon = self.intent_max_horizon
        terminate_th = self.terminate_th
        r_discount = self.r_discount
        running_envs = self.running_envs
        num_envs = self.num_procs
        op_steps = self.op_steps

        recurrent = algo.recurrent
        model = algo.model
        dtype = torch.float

        running_envs_ids = list(range(num_envs))
        env_dones = [False] * num_envs
        env_infos = [dict() for _ in range(num_envs)]
        env_rs = [list() for _ in range(num_envs)]

        running_envs.fill_(True)
        op_steps.fill_(0)

        first_obs = copy.deepcopy(algo.obs)
        obs_stack = [list() for _ in range(num_envs)]

        actions_taken = [list() for _ in range(num_envs)]

        # Allocate options
        first_time = True
        oracle_actions = self._oracle_actions

        # Run max intent horizon steps
        while len(running_envs_ids) > 0:
            crt_obss = []
            crt_ops = []
            for eid in running_envs_ids:
                algo.obs[eid]["prev_option"] = op_ids[eid]
                crt_ops.append(op_ids[eid])
                crt_obss.append(algo.obs[eid])

            preprocessed_obs = algo.preprocess_obss(crt_obss, device=device)

            if recurrent:
                crt_mem = algo.memory[running_envs] * algo.mask.unsqueeze(1)[running_envs]

            if oracle_actions:
                if first_time:
                    first_time = False
                    algo.model.options.generate_solutions(preprocessed_obs)

                # Informed env id option model run
                with torch.no_grad():
                    res_m = model(preprocessed_obs, env_ids=running_envs_ids)
            else:
                with torch.no_grad():
                    if recurrent:
                        res_m = model(preprocessed_obs, memory=crt_mem)
                    else:
                        res_m = model(preprocessed_obs)

            action, act_log_prob, value = res_m["actions"], \
                                          res_m["act_log_probs"], \
                                          res_m["values"]

            # ======================================================================================
            # End trajectory if terminated
            if self.terminate_op_gt:
                if self.terminate_any:
                    op_teriminated = [zzz["collected"] != -1 for zzz in crt_obss]
                else:
                    op_teriminated = [zzz["collected"] == qqq for zzz, qqq in zip(crt_obss, crt_ops)]
            else:
                if self.terminate_any:
                    op_teriminated = (res_m["op_terminations"] > terminate_th).any(dim=1)
                else:
                    op_teriminated = res_m["op_terminate"] > terminate_th

            terminated_envs = []
            for i, (eid, ttt) in enumerate(zip(running_envs_ids, op_teriminated)):
                if ttt and op_steps[eid] > 0:
                    running_envs[eid] = False
                    terminated_envs.append(i)

            if len(terminated_envs) > 0:
                select = torch.ones_like(action).bool()
                for i in terminated_envs[::-1]:
                    running_envs_ids.pop(i)
                    select[i] = False

                action = action[select]
                if recurrent:
                    res_m["memory"] = res_m["memory"][select]

            if len(action) <= 0:
                break
            # ======================================================================================

            send_actions = algo._process_actions_for_step(action, res_m)
            for eid, aaa in zip(running_envs_ids, send_actions):
                actions_taken[eid].append(aaa)

            obs, reward, done, info = algo.env.step(send_actions, running_envs_ids)

            obs = self.post_process(obs)

            if not algo._protect_env_data:
                obs, reward, done, info = list(obs), list(reward), list(done), list(info)

            dones = torch.tensor(done, device=device, dtype=dtype)
            algo.mask[running_envs] = torch.tensor(1.) - dones

            if recurrent:
                algo.memory[running_envs] = res_m["memory"]

            op_steps[running_envs] += 1

            terminated_envs = []

            for i, (eid, ooo, rrr, ddd) in enumerate(zip(running_envs_ids, obs, reward, done)):
                ooo["prev_option"] = op_ids[eid]
                algo.obs[eid] = ooo
                env_dones[eid] = ddd
                env_rs[eid].append(rrr)
                env_infos[eid] = info[i]
                if ddd or op_steps[eid] > intent_max_horizon:
                    running_envs[eid] = False
                    terminated_envs.append(i)

                obs_stack[eid].append(ooo)

            for i in terminated_envs[::-1]:
                running_envs_ids.pop(i)

            self._crt_step += 1

        if recurrent:
            algo.memory.zero_()

        # Calculated discounted r
        for i in range(num_envs):
            r_d = env_rs[i][-1]
            env_rs[i] = env_rs[i][:-1]
            for r in env_rs[i][::-1]:
                r_d = r + r_d * r_discount
            env_rs[i] = r_d

        obs = self.post_process(algo.obs)
        obs = self.add_interest(obs)

        self._last_env_step_cnt += op_steps.sum().item()
        self._total_frames += op_steps.sum().item()

        # change task
        if self.multitask_frame_steps > 0:
            if (self.multitask_frame_steps + self.multitask_last_frame) < self._total_frames:
                algo.env.step([999] * algo.env.num_procs)
                self.multitask_last_frame += self.multitask_frame_steps

        self.step_r_stack = list()
        self.step_done_stack = list()

        for einfo, obsse in zip(env_infos, obs_stack):
            einfo["extra_obs"] = obsse
        return cp(obs), cp(env_rs), cp(env_dones), cp(env_infos)

    def fetch_crt_env_steps(self):
        crt_steps = self._last_env_step_cnt
        self._last_env_step_cnt = 0
        return crt_steps

    def reset(self):
        self._crt_step = 0
        self.op_steps.fill_(0)

        obs = self.algo.obs
        obs = self.post_process(obs)
        obs = self.add_interest(obs)

        return cp(obs)

    def close(self):
        algo = self.algo
        algo.env.close_procs()

        if hasattr(algo.eval_envs, "close_procs"):
            algo.eval_envs.close_procs()

    def post_process(self, obs):
        # Calculated discounted r
        op_steps = self.op_steps

        for i in range(len(obs)):
            obs[i]["num_steps"] = op_steps[i].item()

        return obs

    def add_interest(self, obs):
        if self.modulate_policy == "nan":
            return obs

        if self.gt_affordable:
            for i in range(len(obs)):
                if "affordable" in obs[i]:
                    aff = obs[i]["affordable"] + EPS
                else:
                    aff = np.array(obs[i]["available_obj"]) + EPS

                aff_sum = aff.sum()
                aff /= aff_sum
                obs[i]["interest"] = obs[i]["op_aff"] = aff
            return obs

        algo = self.algo
        with torch.no_grad():
            preprocessed_obs = algo.preprocess_obss(obs, device=self.device)
            rrr = algo.model.get_interest(preprocessed_obs)
        interest = rrr["interest"]
        op_aff = rrr["op_aff"]
        interest = interest.data.cpu().numpy()
        op_aff = op_aff.data.cpu().numpy()

        if self.modulate_policy == "affordance":
            interest = (op_aff > self.affordance_th).astype(np.float) + EPS
            interest_sum = interest.sum(axis=1)
            interest /= np.repeat(interest_sum[:, np.newaxis], interest.shape[1], axis=1)

        for i in range(len(obs)):
            obs[i]["interest"] = interest[i]
            obs[i]["op_aff"] = op_aff[i]

        return obs


def pre_process_cfg(full_args: Namespace):
    add_to_cfg(full_args, MAIN_CFG_ARGS, "out_dir", full_args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and full_args.main.use_gpu else "cpu")

    full_args.device = device
    add_to_cfg(full_args, MAIN_CFG_ARGS, "device", device)

    # Adjust seed according to run_id if seed set to random
    if full_args.main.seed == 0:
        full_args.main.seed = full_args.run_id + 1

    if hasattr(full_args.main, "modulate_policy"):
        if full_args.main.modulate_policy == "nan":
            full_args.model.use_interest = False
        else:
            mp = full_args.main.modulate_policy
            assert mp == "affordance" or mp == "interest", f"Unknown type of policy modulation {mp}"
            full_args.model.use_interest = True

    return full_args


def post_process_cfg(full_args: Namespace, env: gym.Env):
    return full_args


def run(full_args: Namespace, return_models: bool = False, load_model_id: int = None):

    # ==============================================================================================
    # -- Config
    full_args = pre_process_cfg(full_args)

    args = full_args.main
    model_args = full_args.model

    device = full_args.device

    max_eprews = args.max_eprews
    max_eprews_window = getattr(args, "max_eprews_window", -1)
    main_r_key = getattr(args, "main_r_key", "return_per_episode")

    out_dir = getattr(args, "model_dir", full_args.out_dir)

    # ==============================================================================================
    # -- Load pre-trained Options

    # If CFG is a folder than we choose the same run_id for cfg
    if os.path.isdir(args.options_model_cfg):
        crt_exp_run_id = full_args.run_id
        checkpoint_cfg = f"{args.options_model_cfg}/{crt_exp_run_id}/cfg.yaml"
        assert os.path.isfile(checkpoint_cfg), f"No checkpoint cfg matching {checkpoint_cfg}"
    else:
        checkpoint_cfg = args.options_model_cfg

    # Dummy config to load liftoff
    sys.argv = ['manual_control.py', checkpoint_cfg, '--session-id', '1']

    op_cfg = parse_opts(check_out_dir=False)
    op_cfg = pre_process_cfg(op_cfg)

    model_paths = glob.glob(f"{op_cfg.out_dir}/checkpoints/training_data*")
    model_ids = [int(re.findall("training_data_(.*).pt", x)[0]) for x in model_paths]
    model_ids.sort()

    if args.options_model_id < 0:
        model_selection = model_ids[args.options_model_id]
    else:
        assert args.options_model_id in model_ids, "Model ID doesn't exist"
        model_selection = args.options_model_id

    multitask = False
    load_option_envs = True
    op_cfg.main.plot = False
    multitask_frame_steps = 0
    gt_affordable = False

    if "BlockedUnlock" in op_cfg.main.env:
        op_cfg.env_cfg.env_args.with_reward = True
        op_cfg.env_cfg.env_args.full_task = True
        op_cfg.env_cfg.env_args.reset_on_intent = False
        op_cfg.agent.send_intent_to_env = False
    elif op_cfg.main.env.startswith("MiniGrid-MultiObject"):
        op_cfg.env_cfg.env_args.full_task = True
        if hasattr(args, "terminate_op_gt"):
            op_cfg.agent.terminate_op_gt = args.terminate_op_gt
        if hasattr(args, "terminate_th"):
            op_cfg.agent.terminate_th = args.terminate_th
        if hasattr(args, "multitask") and args.multitask:
            class_name = getattr(args, "multitask_class", "MultiTaskMultiObj")
            op_cfg.env_cfg.wrapper = [class_name] + op_cfg.env_cfg.wrapper
    elif "MonsterKong" in op_cfg.main.env:
        op_cfg.env_cfg.env_args.random_level = False
        if hasattr(args, "terminate_op_gt"):
            op_cfg.agent.terminate_op_gt = args.terminate_op_gt
        gt_affordable = getattr(args, "gt_affordable", False)
    else:
        op_cfg.env_cfg.env_args.fake_goal = False  # Activate full task
        op_cfg.env_cfg.env_args.reward_room = True  # Activate full task
        op_cfg.env_cfg.env_args.agent_pos = args.env_agent_start_pos
        op_cfg.env_cfg.env_args.goal_pos = args.env_goal_pos
        op_cfg.env_cfg.env_args.goal_rooms = args.goal_rooms
        op_cfg.env_cfg.env_args.goal_rooms = args.goal_rooms

        if hasattr(args, "close_doors_trials"):
            op_cfg.env_cfg.env_args.close_doors_trials = args.close_doors_trials

        multitask_frame_steps = 0
        if hasattr(args, "multitask"):
            op_cfg.env_cfg.env_args.multitask = args.multitask

        gt_affordable = getattr(args, "gt_affordable", False)
        if gt_affordable:
            # configure for ground truth affordances
            wrapper_id = op_cfg.env_cfg.wrapper.index("ExtendRoomFullyObs")
            op_cfg.env_cfg.wrapper[wrapper_id] = "ExtendRoomFullyObsExtra"

    if hasattr(args, "multitask"):
        # TODO hardcoded frame conversion
        if args.multitask:
            multitask_frame_steps = args.multitask_update_step * args.multitask_update_size
        else:
            multitask_frame_steps = 0

    if hasattr(args, "intent_max_horizon"):
        op_cfg.agent.intent_max_horizon = args.intent_max_horizon
    if hasattr(args, "procs"):
        op_cfg.main.procs = args.procs
    if hasattr(args, "max_episode_steps"):
        op_cfg.env_cfg.max_episode_steps = args.max_episode_steps
    if hasattr(args, "option_discount"):
        op_cfg.agent.option_discount = args.option_discount

    offset_frame_limit = getattr(args, "offset_frame_limit", False)

    op_cfg.agent.parallel_env_class = "ParallelEnvWithLastObsIndex"

    oracle_actions = getattr(args, "oracle_actions", False)

    op_agent = OptionsAgent(op_cfg, model_selection, full_args,
                            multitask_frame_steps=multitask_frame_steps,
                            gt_affordable=gt_affordable, oracle_actions=oracle_actions)

    if hasattr(args, "interest_temp"):
        op_agent.algo.model.interest_temp = args.interest_temp

    # -- Copy some cfg from the loaded model so we can filter plots
    full_args.main.ckpt_affordance_train_epochs = op_cfg.agent.affordance_train_epochs
    full_args.main.ckpt_aff_loss_balance = op_cfg.agent.aff_loss_balance
    full_args.main.ckpt_aff_criterion_type = op_cfg.agent.aff_criterion.aff_criterion_type

    # Define action space for policy over options
    num_intents = Namespace()
    num_intents.n = op_cfg.agent.intent_size

    # ==============================================================================================
    # -- Define logger, CSV writer, plot logger
    plot_x_axis = getattr(args, "plot_x_axis", "frames")
    plot_project = getattr(args, "plot_project", "affordanceoptions")
    experiment_name = f"{full_args.full_title}_{full_args.run_id}"

    stats_window_size = args.stats_window_size

    log_out_dir = None if return_models else out_dir
    logger_master = LoggerMaster(log_out_dir, plot=args.plot, plot_x=plot_x_axis,
                                 stats_window_size=stats_window_size,
                                 plot_project=plot_project, experiment=experiment_name,
                                 cfg=dict(flatten_cfg(full_args)))
    logger = logger_master.get_logger()

    # Log command and all script arguments
    logger.info("{}\n".format(" ".join(sys.argv)))
    logger.info("{}\n".format(args))

    # ==============================================================================================
    # -- Set seed for all randomness sources

    utils.seed(args.seed)

    # ==============================================================================================
    # -- Prepare environments (Train & eval)

    envs, eval_envs = [], []

    num_eval_envs = args.num_eval_envs
    eval_episodes = getattr(args, "eval_episodes", 0)
    wrapper_methods = getattr(op_cfg.env_cfg, "wrapper", None)

    env_cfg = op_cfg.env_cfg

    # Load envs from option agent
    envs = op_agent if load_option_envs else op_agent.algo.env
    first_env = op_agent.first_env

    # -- Define obss preprocessor
    obs_space, preprocess_obss = get_obss_pre_processor(
        op_cfg.main.env, first_env.observation_space, out_dir, env_cfg
    )

    full_args = post_process_cfg(full_args, first_env)  # For custom use (e.g. add info from env)

    # ==============================================================================================
    # -- Load training status

    saver = save_training.SaveData(out_dir, save_best=args.save_best, save_all=args.save_all)
    model, agent_data, other_data = None, dict(), None

    try:
        # Continue from last point
        model, agent_data, other_data = saver.load_training_data(best=False, index=load_model_id)
        logger.info("Training data exists & loaded successfully\n")
    except OSError:
        logger.info("Could not load training data\n")

    num_frames_offset = 0
    if other_data is None or len(other_data) == 0:
        status = Namespace()
        status.num_frames = 0
        status.update = 1

        # Offset frame count by pre-trained model frames
        num_frames_offset = op_agent.saver._loaded_status.num_frames
        status.num_frames += num_frames_offset

        if offset_frame_limit:
            args.frames = num_frames_offset + args.frames

    else:
        status = Namespace()
        status.__dict__.update(other_data)

    # ==============================================================================================
    # -- Load Model
    if model is None:
        model = get_model(model_args, obs_space, num_intents)
        logger.info(f"Model [{model_args.name}] successfully created\n")

        # Print Model info
        logger.info("{}\n".format(model))

    model.to(device)

    logger.info("Device used: {}\n".format(device))

    # ==============================================================================================
    # -- Load Agent

    algo = get_agent(full_args.agent, envs, model,
                     preprocess_obss=preprocess_obss,
                     eval_envs=eval_envs, eval_episodes=eval_episodes)

    # Load agent data
    algo.load_checkpoint(agent_data)

    has_evaluator = algo.has_evaluator and num_eval_envs > 0

    # Used for e.g. loading training
    if return_models:
        return algo, model, envs, saver

    # ==============================================================================================
    # -- Register logging headers from algorithm & toggle print

    toggle_print = getattr(full_args, "toggle_print", [])
    toggle_plot = getattr(full_args, "toggle_plot", [])

    train_header_config, eval_header_config = algo.get_logs_config()

    logger_master.set_header(status.update, train_header_config)
    logger_master.toggle_print(toggle_print)
    logger_master.toggle_plot(toggle_plot)

    logger_eval = None  # type: LoggerMaster
    if has_evaluator:
        logger_eval = LoggerMaster(log_out_dir, plot=args.plot, plot_x=plot_x_axis,
                                   stats_window_size=stats_window_size, logger=logger,
                                   plotter=logger_master.plotter, csv_file_name="eval.csv",
                                   plot_project=plot_project, experiment=experiment_name)
        logger_eval.set_header(status.update, eval_header_config)
        logger_eval.toggle_print(toggle_print)
        logger_eval.toggle_plot(toggle_plot)

    # ==============================================================================================
    # -- Training loop

    prev_rewards = []

    while status.num_frames < args.frames:

        # ================================================================
        # SMDP Memorise start of algorithm
        op_agent.step = op_agent.standard_step

        # ================================================================

        # -- Update model parameters & return logs
        logs = algo.update_parameters()

        # Consider environment steps (vs SMDP level steps)
        status.num_frames += op_agent.fetch_crt_env_steps()

        logs["num_frames"] = [status.num_frames]
        logs["num_frames_no_off"] = [status.num_frames - num_frames_offset]

        # -- Update status with crt logs
        update_num_frames = logs["num_frames"][0]

        # -- Register logs for each step
        logger_master.register_log(status.update, update_num_frames, logs)

        crt_r = logger_master.header.get_win_value(main_r_key, default=0)
        prev_rewards.append(crt_r)

        if has_evaluator and status.update % args.eval_interval == 0:
            eval_logs = algo.evaluate(eval_key=main_r_key)
            eval_logs.update(vars(status))
            logger_eval.register_log(status.update, update_num_frames, eval_logs)

        # -- Print logs
        if status.update % args.log_interval == 0:
            logger_master.log_logs()

        # -- Save checkpoints
        if args.save_interval > 0 and status.update % args.save_interval == 0:
            if hasattr(preprocess_obss, "vocab"):
                preprocess_obss.vocab.save()

            saver.save_training_data(model, algo.get_checkpoint(), crt_r, other=vars(status))
            logger.info("Checkpoint successfully saved")

        # -- Early stop when reaching specific reward mean for a given window
        if max_eprews_window > 0:
            check_rew = np.mean(prev_rewards[-max_eprews_window:])
            if len(prev_rewards) > max_eprews_window and check_rew > max_eprews:
                logger.info(f"Reached mean return {max_eprews} for a window of "
                            f"{max_eprews_window} steps")
                exit()

        status.update += 1
    exit()


if __name__ == "__main__":
    run(parse_opts())
