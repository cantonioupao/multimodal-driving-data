``tornado.gen`` --- Generator-based coroutines
==============================================

.. testsetup::

   from tornado.web import *
   from tornado import gen

.. automodule:: tornado.gen

   Decorators
   ----------

   .. autofunction:: coroutine

   .. autofunction:: engine

   Utility functions
   -----------------

   .. autoexception:: Return

   .. autofunction:: with_timeout

   .. autofunction:: sleep

   .. autodata:: moment
      :annotation:

   .. autoclass:: WaitIterator
      :members:

   .. autofunction:: multi

   .. autofunction:: multi_future

   .. autofunction:: convert_yielded

   .. autofunction:: maybe_future

   .. autofunction:: is_coroutine_function

   Legacy interface
   ----------------

   Before support for `Futures <.Future>` was introduced in Tornado 3.0,
   coroutines used subclasses of `YieldPoint` in their ``yield`` expressions.
   These classes are still supported but should generally not be used
   except for compatibility with older interfaces. None of these classes
   are compatible with native (``await``-based) coroutines.

   .. autoclass:: YieldPoint
      :members:

   .. autoclass:: Callback

   .. autoclass:: Wait

   .. autoclass:: WaitAll

   .. autoclass:: MultiYieldPoint

   .. autofunction:: Task

   .. class:: Arguments

      The result of a `Task` or `Wait` whose callback had more than one
      argument (or keyword arguments).

      The `Arguments` object is a `collections.namedtuple` and can be
      used either as a tuple ``(args, kwargs)`` or an object with attributes
      ``args`` and ``kwargs``.

      .. deprecated:: 5.1

         This class will be removed in 6.0.
