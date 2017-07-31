from collections import defaultdict
import visdom
from .plugin import Plugin
from .logger import Logger

import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib


class BaseVisdomLogger(Logger):
    ''' 
        The base class for logging output to Visdom. 

        ***THIS CLASS IS ABSTRACT AND MUST BE SUBCLASSED***
    '''
    _viz = visdom.Visdom()

    @property
    def viz(self):
        return type(self)._viz

    def __init__(self, fields, interval=None, win=None, env=None, opts={}):
        super(BaseVisdomLogger, self).__init__(fields, interval)
        self.win = win
        self.env = env
        self.opts = opts

    def log(self, *args):
        raise NotImplementedError("log not implemented for BaseVisdomLogger, which is an abstract class.")

    def _viz_prototype(self, vis_fn):
        def _viz_logger(*args):
            self.win = vis_fn(args, 
                    win=self.win,
                    env=self.env,
                    opts=self.opts)
        return _viz_logger
# class VisdomImageLogger(BaseVisdomLogger):

class VisdomPlotLogger(BaseVisdomLogger):
    
    def __init__(self, fields, interval=None, opts={}):
        '''
            opts: dict of opts. May specify the plot type with 
                    plot_type \in {SCATTER, LINE}

            Examples::
                >>> train = Trainer(model, criterion, optimizer, dataset)
                >>> progress_m = ProgressMonitor()
                >>> scatter_logger = VisdomScatterLogger(["progress.samples_used", "progress.percent"], [(2, 'iteration')])
                >>> train.register_plugin(progress_m)
                >>> train.register_plugin(scatter_logger)
        '''
        super(VisdomPlotLogger, self).__init__(fields, interval)
        valid_plot_types = {
            "SCATTER": self.viz.scatter, 
            "LINE": self.viz.line }

        # Set chart type
        if 'plot_type' in self.opts:
            if self.opts['plot_type'] not in valid_plot_types.keys():
                raise ValueError("plot_type \'{}\' not found. Must be one of {}".format(
                    self.opts['plot_type'], valid_plot_types.keys()))
            self.chart = valid_plot_types[self.opts['plot_type']]
        else:
            self.chart = self.viz.scatter

    def log(self, *args):
        if self.win is not None:
            self.viz.updateTrace(
                X=np.array([args[0]]),
                Y=np.array([args[1]]),
                win=self.win,
                env=self.env,
                opts=self.opts)
        else:
            self.win = self.viz.scatter(
                X=np.array([args]),
                win=self.win,
                env=self.env,
                opts=self.opts)

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            results.append(stat)
        self.log(*results)


class VisdomTextLogger(BaseVisdomLogger):
    '''
        Creates a text window in visdom and logs output to it. 
    '''
    valid_update_types = ['REPLACE', 'APPEND']

    def __init__(self, fields, interval=None, opts={}, update_type=valid_update_types[0]):
        super(VisdomTextLogger, self).__init__(fields, interval)
        self.text = ''

        if update_type not in self.valid_update_types:
            raise ValueError("update type '{}' not found. Must be one of {}".format(update_type, self.valid_update_types))
        self.update_type = update_type

        self.viz_logger = self._viz_prototype(self.viz.text)

        # Use specific window
        if self.win is None:
            self.win = self.viz.text(
                self.text, 
                win=self.win,
                env=self.env,
                opts=self.opts)
            print("Win: ", self.win)

    def log(self, *args):
        text = args[0]
        if self.update_type == 'APPEND':
            self.text = "<br>".join([self.text, text])
        else:
            self.text = text
        self.viz_logger([self.text])
        # self.viz.text(
        #     self.text, 
        #     win=self.win,
        #     env=self.env,
        #     opts=self.opts)
        

class TestVisdomLogger(BaseVisdomLogger):
    
    def __init__(self, fields, interval=None):
        super(VisdomLogger, self).__init__(fields, interval)

    def log(self, *args):
        viz = self.viz
        textwindow = viz.text('Hello World!')

        # video demo:
        try:
            video = np.empty([256, 250, 250, 3], dtype=np.uint8)
            for n in range(256):
                video[n, :, :, :].fill(n)
            viz.video(tensor=video)

            # video demo: download video from http://media.w3.org/2010/05/sintel/trailer.ogv
            video_url = 'http://media.w3.org/2010/05/sintel/trailer.ogv'
            # linux
            if _platform == "linux" or _platform == "linux2":
                videofile = '/home/%s/trailer.ogv' % getpass.getuser()
            # MAC OS X
            elif _platform == "darwin":
                videofile = '/Users/%s/trailer.ogv' % getpass.getuser()
            # download video
            urllib.request.urlretrieve(video_url, videofile)

            if os.path.isfile(videofile):
                viz.video(videofile=videofile)
        except ImportError:
            print('Skipped video example')


        # image demo
        viz.image(
            np.random.rand(3, 512, 256),
            opts=dict(title='Random!', caption='How random.'),
        )

        # grid of images
        viz.images(
            np.random.randn(20, 3, 64, 64),
            opts=dict(title='Random images', caption='How random.')
        )

        # scatter plots
        Y = np.random.rand(100)
        viz.scatter(
            X=np.random.rand(100, 2),
            Y=(Y[Y > 0] + 1.5).astype(int),
            opts=dict(
                legend=['Apples', 'Pears'],
                xtickmin=-5,
                xtickmax=5,
                xtickstep=0.5,
                ytickmin=-5,
                ytickmax=5,
                ytickstep=0.5,
                markersymbol='cross-thin-open',
            ),
        )

        viz.scatter(
            X=np.random.rand(100, 3),
            Y=(Y + 1.5).astype(int),
            opts=dict(
                legend=['Men', 'Women'],
                markersize=5,
            )
        )

        # 2D scatterplot with custom intensities (red channel)
        viz.scatter(
            X=np.random.rand(255, 2),
            Y=(np.random.rand(255) + 1.5).astype(int),
            opts=dict(
                markersize=10,
                markercolor=np.random.randint(0, 255, (2, 3,)),
            ),
        )

        # 2D scatter plot with custom colors per label:
        viz.scatter(
            X=np.random.rand(255, 2),
            Y=(np.random.randn(255) > 0) + 1,
            opts=dict(
                markersize=10,
                markercolor=np.floor(np.random.random((2, 3)) * 255),
            ),
        )

        win = viz.scatter(
            X=np.random.rand(255, 2),
            opts=dict(
                markersize=10,
                markercolor=np.random.randint(0, 255, (255, 3,)),
            ),
        )

        # add new trace to scatter plot
        viz.updateTrace(
            X=np.random.rand(255),
            Y=np.random.rand(255),
            win=win,
            name='new_trace',
        )


        # bar plots
        viz.bar(X=np.random.rand(20))
        viz.bar(
            X=np.abs(np.random.rand(5, 3)),
            opts=dict(
                stacked=True,
                legend=['Facebook', 'Google', 'Twitter'],
                rownames=['2012', '2013', '2014', '2015', '2016']
            )
        )
        viz.bar(
            X=np.random.rand(20, 3),
            opts=dict(
                stacked=False,
                legend=['The Netherlands', 'France', 'United States']
            )
        )

        # histogram
        viz.histogram(X=np.random.rand(10000), opts=dict(numbins=20))

        # heatmap
        viz.heatmap(
            X=np.outer(np.arange(1, 6), np.arange(1, 11)),
            opts=dict(
                columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
                colormap='Electric',
            )
        )

        # contour
        x = np.tile(np.arange(1, 101), (100, 1))
        y = x.transpose()
        X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
        viz.contour(X=X, opts=dict(colormap='Viridis'))

        # surface
        viz.surf(X=X, opts=dict(colormap='Hot'))

        # line plots
        viz.line(Y=np.random.rand(10))

        Y = np.linspace(-5, 5, 100)
        viz.line(
            Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
            X=np.column_stack((Y, Y)),
            opts=dict(markers=False),
        )

        # line updates
        win = viz.line(
            X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
            Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
        )
        viz.line(
            X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
            Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
            win=win,
            update='append'
        )
        viz.updateTrace(
            X=np.arange(21, 30),
            Y=np.arange(1, 10),
            win=win,
            name='2'
        )
        viz.updateTrace(
            X=np.arange(1, 10),
            Y=np.arange(11, 20),
            win=win,
            name='4'
        )

        Y = np.linspace(0, 4, 200)
        win = viz.line(
            Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
            X=np.column_stack((Y, Y)),
            opts=dict(
                fillarea=True,
                legend=False,
                width=400,
                height=400,
                xlabel='Time',
                ylabel='Volume',
                ytype='log',
                title='Stacked area plot',
                marginleft=30,
                marginright=30,
                marginbottom=80,
                margintop=30,
            ),
        )

        # boxplot
        X = np.random.rand(100, 2)
        X[:, 1] += 2
        viz.boxplot(
            X=X,
            opts=dict(legend=['Men', 'Women'])
        )

        # stemplot
        Y = np.linspace(0, 2 * math.pi, 70)
        X = np.column_stack((np.sin(Y), np.cos(Y)))
        viz.stem(
            X=X,
            Y=Y,
            opts=dict(legend=['Sine', 'Cosine'])
        )

        # quiver plot
        X = np.arange(0, 2.1, .2)
        Y = np.arange(0, 2.1, .2)
        X = np.broadcast_to(np.expand_dims(X, axis=1), (len(X), len(X)))
        Y = np.broadcast_to(np.expand_dims(Y, axis=0), (len(Y), len(Y)))
        U = np.multiply(np.cos(X), Y)
        V = np.multiply(np.sin(X), Y)
        viz.quiver(
            X=U,
            Y=V,
            opts=dict(normalize=0.9),
        )

        # pie chart
        X = np.asarray([19, 26, 55])
        viz.pie(
            X=X,
            opts=dict(legend=['Residential', 'Non-Residential', 'Utility'])
        )

        # mesh plot
        x = [0, 0, 1, 1, 0, 0, 1, 1]
        y = [0, 1, 1, 0, 0, 1, 1, 0]
        z = [0, 0, 0, 0, 1, 1, 1, 1]
        X = np.c_[x, y, z]
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        Y = np.c_[i, j, k]
        viz.mesh(X=X, Y=Y, opts=dict(opacity=0.5))

        # SVG plotting
        svgstr = """
        <svg height="300" width="300">
        <ellipse cx="80" cy="80" rx="50" ry="30"
        style="fill:red;stroke:purple;stroke-width:2" />
        Sorry, your browser does not support inline SVG.
        </svg>
        """
        viz.svg(
            svgstr=svgstr,
            opts=dict(title='Example of SVG Rendering')
        )

        # close text window:
        viz.close(win=textwindow)

        # PyTorch tensor
        try:
            import torch
            viz.line(Y=torch.Tensor([[0., 0.], [1., 1.]]))
        except ImportError:
            print('Skipped PyTorch example')