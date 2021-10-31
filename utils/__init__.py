from typing import List
from gym import spaces
from argparse import Namespace

from . import gym_wrappers
from . import obs_preprocessors
from . import intent_completion


def get_wrappers(wrappers: List[str]):
    # Get env wrappers - must be a list of elements
    if wrappers is None:
        def idem(x):
            return x

        env_wrapper = idem
    else:
        env_wrappers = [getattr(gym_wrappers, w_p) for w_p in wrappers]

        def env_wrapp(w_env):
            for wrapper in env_wrappers[::-1]:
                w_env = wrapper(w_env)
            return w_env

        env_wrapper = env_wrapp
    return env_wrapper


def get_obss_pre_processor(env_name: str, obs_space: spaces, out_dir: str, cfg: Namespace):
    obss_preprocessor = getattr(cfg, "obss_preprocessor", "process_simple")
    obss_preprocessor = getattr(obs_preprocessors, obss_preprocessor)
    return obss_preprocessor(env_name, obs_space, out_dir, cfg)


def get_intent_completion_method(intent_completion_method: str, cfg: Namespace) -> \
        intent_completion.IntentCompletionBase:
    cls = getattr(intent_completion, intent_completion_method)
    return cls(cfg)
