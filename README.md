# Add a TF-slim like framework to PyTorch, to enable rapid research.
### And bring the TensorBoard-like power of Visdom to PyTorch!

## This code is now integrated into [TNT](https://github.com/pytorch/tnt)
TNT is the 'official' framework for PyTorch and is expected to be merged into PyTorch itself. I'll push updates with new features directly to TNT.

---

## Sample
Visdom is a powerful and flexible platform for visualizing data, from FB. It fulfills much the same role as tensorboard, and is simple to use. 
![Example](https://user-images.githubusercontent.com/5157485/28799619-2bebef8c-75fe-11e7-898d-202a6c6d3239.png)


This repo contains a collection of tools for easily logging to Visdom, and for reusing these tools across different projects. It will also contain nice tools for training models. The vizualization above is logged and also saved periodically (easily adjustable with a parameter) using the code below:
```
    # Plugins produce statistics
    progress_plug = ProgressMonitor()
    random_plug = RandomMonitor(10000)
    image_plug = ConstantMonitor(data.coffee().swapaxes(0,2).swapaxes(1,2), "image")

    # Loggers are a special type of plugin which, surprise, logs the stats
    logger = Logger(["progress"], [(2, 'iteration')])
    text_logger    = VisdomTextLogger(["progress"], [(2, 'iteration')], update_type='APPEND',
                        env=env, opts=dict(title='Example logging'))
    scatter_logger = VisdomPlotLogger('scatter', ["progress.samples_used", "progress.percent"], [(1, 'iteration')],
                        env=env, opts=dict(title='Percent Done vs Samples Used'))
    hist_logger    = VisdomLogger('histogram', ["random.data"], [(2, 'iteration')],
                        env=env, opts=dict(title='Random!', numbins=20))
    image_logger   = VisdomLogger('image', ["image.data"], [(2, 'iteration')], env=env)


    # Create a saver
    saver = VisdomSaver(envs=[env])

    # Register the plugins with the trainer
    train.register_plugin(progress_plug)
    train.register_plugin(random_plug)
    train.register_plugin(image_plug)

    train.register_plugin(logger)
    train.register_plugin(text_logger)
    train.register_plugin(scatter_logger)
    train.register_plugin(hist_logger)
    train.register_plugin(image_logger)

    train.register_plugin(saver)
```


---
## References
 The trainer and plugin framework is taken, with slight modifications, from the main PyTorch branch. Ideally, the functionality from this repo can be pulled back into PyTorch so it is more easily available, and can be used with some existing great libraries like 
 - [TNT](http://github.com/PyTorch/tnt)
 - [TorchSample](http://github.com/ncullen93/torchsample)
 
 Also consider [Inferno](https://github.com/nasimrahaman/inferno) which is new and under heavy active development.
