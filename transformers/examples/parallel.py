from typing import List, Callable
from functools import partial
from concurrent import futures
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import OrderedDict
@contextmanager
def NoPoolExecutor():
  """
  Provides the same interface as ThreadPoolExecutor and ProcessPoolExecutor,
  but does not do any threading / multiprocessing.
  Submitting to NoPoolExecutor returns a function wrapper that acts like a Future,
  but the function is executed synchronously when .result() is called.
  """
  class NoPoolExecutorInner:
    class NoFuture:
      def __init__(self, fn, **kwargs):
        self.fn = partial(fn, **kwargs)
      def result(self):
        return self.fn()
    def submit(self, fn, **kwargs):
      return NoPoolExecutorInner.NoFuture(fn, **kwargs)
    def shutdown(self, *args, **kwargs):
      pass
  yield NoPoolExecutorInner()
def parallelized_iterator(
  fn: Callable,
  kwargs_list: List[dict],
  tags_list: List[object] = None,
  scheduler: str = "threads",
  max_workers: int = 0,
  progress: bool = True,
  progress_auto: bool = True,
  progress_desc: Callable[[int], str] = None,
  progress_position: int = 0,
  keep_order: bool = False,
  cleanup: Callable[[ThreadPoolExecutor], None] = None
):
  """
  Args:
    fn: function to execute
    kwargs_list: arguments for each future call
    tags_list: return argument associated to each future
    scheduler: backend for threading. Defaults to "threads".
    max_workers: max workers.
    progress: show progress.
    progress_auto: show progress with auto.
    progress_desc: function that produces a description.
    keep_order: return results in order of kwargs
    cleanup: function that handles exception cleanup.
  Yields:
    for each kwargs in kwargs_list:
      kwargs, fn(**kwargs)
  """
  if not kwargs_list:
    return []
  if not isinstance(kwargs_list[0], dict):
    raise ValueError("kwargs_list elements must be a dict.")
  if cleanup is None:
    def default_cleanup(executor):
      executor.shutdown(wait=True)
    cleanup = default_cleanup
  if progress_auto:
    from tqdm.auto import tqdm
  else:
    from tqdm import tqdm
  if tags_list is None:
      tags_list = kwargs_list
  if len(tags_list) != len(kwargs_list):
    raise ValueError(
        "Number of tags should match the number of jobs to parallelize.")
  ordered = lambda x: x  # noqa: E731
  if max_workers <= 1:
    pool_executor_cls = NoPoolExecutor
    executor_kwargs = dict()
    as_completed_fn = ordered
  elif scheduler == "processes":
    pool_executor_cls = ProcessPoolExecutor
    executor_kwargs = dict(max_workers=max_workers)
    as_completed_fn = ordered if keep_order else futures.as_completed
  elif scheduler == "threads":
    pool_executor_cls = ThreadPoolExecutor
    executor_kwargs = dict(max_workers=max_workers)
    as_completed_fn = ordered if keep_order else futures.as_completed
  else:
    raise ValueError("Wrong scheduler: %s" % scheduler)
  with pool_executor_cls(**executor_kwargs) as executor:
    futures_dict = OrderedDict(
      (executor.submit(fn, **kwargs), tag)
      for kwargs, tag in zip(kwargs_list, tags_list))
    tqdm_args = {}
    tqdm_args["total"] = len(kwargs_list)
    tqdm_args["disable"] = not progress
    tqdm_args["position"] = progress_position
    with tqdm(**tqdm_args) as bar:
      try:
        if progress_desc is not None:
          bar.set_description(progress_desc(0))
        for i, future in enumerate(as_completed_fn(futures_dict)):
          bar.update(1)
          if progress_desc is not None:
            # strip a right '.' because it doesn't look good with tqdm
            bar.set_description(progress_desc(i + 1).rstrip('.'))
          tag = futures_dict[future]
          yield tag, future.result()
      finally:
        cleanup(executor)
def parallelized(
  fn,
  kwargs_list: List[dict],
  tags_list: List[object] = None,
  scheduler: str = "threads",
  max_workers: int = 0,
  progress: bool = True,
  progress_auto: bool = True,
  progress_desc: Callable[[int], str] = None,
  progress_position: int = 0,
  keep_order: bool = True,
  cleanup: Callable[[ThreadPoolExecutor], None] = None,
):
  results = []
  for _, result in parallelized_iterator(
          fn, kwargs_list, tags_list, scheduler, max_workers,
          progress, progress_auto, progress_desc,
          progress_position, keep_order, cleanup):
    results.append(result)
  return results
