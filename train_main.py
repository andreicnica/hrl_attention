import torch
import sys
import gym

from argparse import Namespace
import numpy as np

try:
    import gym_minigrid
except ImportError:
    pass

from models import get_model
from agents import get_agent
from utils import utils, get_wrappers, save_training, get_obss_pre_processor
from utils.utils import add_to_cfg

from utils.logging_utils import LoggerMaster
from utils.utils import flatten_cfg

MAIN_CFG_ARGS = ["main", "env_cfg", "agent", "model"]


def pre_process_cfg(full_args: Namespace):
    add_to_cfg(full_args, MAIN_CFG_ARGS, "out_dir", full_args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and full_args.main.use_gpu else "cpu")

    full_args.device = device
    add_to_cfg(full_args, MAIN_CFG_ARGS, "device", device)

    # Adjust seed according to run_id if seed set to random
    if full_args.main.seed == 0:
        full_args.main.seed = full_args.run_id + 1

    # HACKY CHECK
    model_args = full_args.model
    env_cfg = full_args.env_cfg

    if hasattr(env_cfg, "env_args"):
        if hasattr(env_cfg.env_args, "send_intent_to_env") and full_args.agent.send_intent_to_env:
            env_cfg.env_args.reset_on_intent = full_args.agent.send_intent_to_env

    if full_args.main.env.startswith("MiniGrid-GridRooms"):
        if full_args.main.env == "MiniGrid-GridRooms9-v0":
            setattr(model_args, "env_max_room", 2)
        elif full_args.main.env == "MiniGrid-GridRooms4-v0":
            setattr(model_args, "env_max_room", 1)

        for wrapper_name in full_args.env_cfg.wrapper:
            if "FreeMove" in wrapper_name:
                assert full_args.env_cfg.max_actions == 4, "Free Move wrappers only have 4 " \
                                                           "actions available"
    elif full_args.main.env.startswith("MiniGrid-MultiObject"):
        intent_size = env_cfg.env_args.task_size * env_cfg.env_args.num_tasks
        setattr(full_args.agent, "intent_size", intent_size)
    elif full_args.main.env.startswith("MiniGrid-"):
        pass
    elif full_args.main.env.startswith("Fetch"):
        pass
    elif full_args.main.env.startswith("MonsterKong"):
        pass
    else:
        raise NotImplementedError

    # Adjust attention type
    attention_type = full_args.main.attention
    if attention_type == "affordance":
        setattr(model_args, "sample_interest", False)
        setattr(model_args, "sample_attention_op", True)
    elif attention_type == "interest":
        setattr(model_args, "sample_interest", True)
        setattr(model_args, "sample_attention_op", True)
    else:
        setattr(model_args, "sample_interest", True)
        setattr(model_args, "sample_attention_op", False)

    setattr(model_args, "terminate_th", getattr(full_args.agent, "terminate_th", None))
    setattr(model_args, "intent_size", getattr(full_args.agent, "intent_size", None))
    setattr(model_args, "num_options", getattr(full_args.agent, "intent_size", None))
    setattr(model_args, "use_affordances", getattr(full_args.agent, "use_affordances", None))
    setattr(model_args, "recurrent", full_args.agent.recurrence > 1)

    if hasattr(full_args.agent, "aff_criterion") and \
            full_args.agent.aff_criterion.aff_criterion_type == "mse":
        model_args.use_sigmoid = False

    return full_args


def post_process_cfg(full_args: Namespace, env: gym.Env):
    return full_args


def run(full_args: Namespace, return_models: bool = False, load_model_id: int = None):

    # ==============================================================================================
    # -- Config
    full_args = pre_process_cfg(full_args)

    args = full_args.main
    model_args = full_args.model
    env_cfg = full_args.env_cfg

    env_name = full_args.main.env  # type: str

    device = full_args.device

    max_eprews = args.max_eprews
    max_eprews_window = getattr(args, "max_eprews_window", -1)
    main_r_key = getattr(args, "main_r_key", "return_per_episode")

    out_dir = getattr(args, "model_dir", full_args.out_dir)
    print(out_dir)

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

    num_envs = args.procs
    num_eval_envs = args.num_eval_envs
    eval_episodes = getattr(args, "eval_episodes", 0)
    wrapper_methods = getattr(full_args.env_cfg, "wrapper", None)

    env_wrapper = get_wrappers(wrapper_methods)

    if env_name.startswith("MiniGrid"):
        from utils.vec_env import get_minigrid_envs

        # Split envs in chunks
        envs, chunk_size = get_minigrid_envs(full_args, env_wrapper, num_envs)
        first_env = envs[0][0]

        # Generate evaluation envs
        if num_eval_envs > 0:
            eval_envs, chunk_size = get_minigrid_envs(full_args, env_wrapper, num_eval_envs)

        logger.info(f"No. of envs / proc: {chunk_size}; No of processes {len(envs[1:])} + Master")
    elif env_name.startswith("Fetch") or env_name.startswith("MonsterKong"):
        from utils.vec_env import get_minigrid_envs

        # Split envs in chunks
        envs, chunk_size = get_minigrid_envs(full_args, env_wrapper, num_envs)
        first_env = envs[0][0]

        # Generate evaluation envs
        if num_eval_envs > 0:
            eval_envs, chunk_size = get_minigrid_envs(full_args, env_wrapper, num_eval_envs)
        logger.info(f"No. of envs / proc: {chunk_size}; No of processes {len(envs[1:])} + Master")
    else:
        raise NotImplementedError

    # -- Define obss preprocessor
    obs_space, preprocess_obss = get_obss_pre_processor(
        args.env, first_env.observation_space, out_dir, env_cfg
    )

    full_args = post_process_cfg(full_args, first_env)  # For custom use (e.g. add info from env)

    multitask = False
    if hasattr(full_args.main, "multitask"):
        multitask = full_args.main.multitask
        multitask_upd = full_args.main.multitask_update_step

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

    if other_data is None or len(other_data) == 0:
        status = Namespace()
        status.num_frames = 0
        status.update = 1
    else:
        status = Namespace()
        status.__dict__.update(other_data)
        saver._loaded_status = status

    # ==============================================================================================
    # -- Load Model

    if model is None:
        model = get_model(model_args, obs_space, first_env.action_space)
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
    switch_sample_aff = getattr(args, "switch_sample_aff", 0)

    # torch.autograd.set_detect_anomaly(True)
    while status.num_frames < args.frames:

        # change task
        if multitask and status.update % multitask_upd == 0:
            algo.env.step([999] * algo.env.num_procs)

        # -- Update model parameters & return logs
        logs = algo.update_parameters()

        # -- Update status with crt logs
        update_num_frames = logs["num_frames"][0]
        status.num_frames += update_num_frames
        logs["num_frames"] = [status.num_frames]
        logs["num_frames_no_off"] = [status.num_frames]

        # -- Register logs for each step
        logger_master.register_log(status.update, update_num_frames, logs)

        crt_r = logger_master.header.get_win_value(main_r_key, default=0)
        prev_rewards.append(crt_r)

        if has_evaluator and status.update % args.eval_interval == 0:
            eval_logs = algo.evaluate(eval_key=main_r_key)
            eval_logs.update({"num_frames": [status.num_frames],
                              "num_frames_no_off": [status.num_frames]})
            logger_eval.register_log(status.update, update_num_frames, eval_logs)
            logger_eval.log_logs()

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


if __name__ == "__main__":
    from utils.utils import parse_opts
    run(parse_opts())
