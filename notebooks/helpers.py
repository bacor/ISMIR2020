# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2020 Bas Cornelissen
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.text import Text
from seaborn.matrix import _HeatMapper
from seaborn.matrix import relative_luminance
import numpy as np

def cm2inch(*args):
    return list(map(lambda x: x/2.54, args))

def title(text, **kwargs):
    plt.title(text, ha='left', x=0, **kwargs)

def highlight_highest_scores(df, ax=None, axis=0, tol=1e-6, line_length=.4, 
    line_y=.25, **line_kwargs):
    """Highlights the highest scores in a heatmap by making them boldface
    and drawing a line underneath

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe as it is fed to the sns.heatmap
    ax : axis, optional
        The axis, defaults to the current axis  
    axis : int, optional
        The axis along which to search for the maximum, by default 0
    tol : int, optional
        The tolerance, by default 1e-6
    line_length : float, optional
        Length of the line, by default .4
    line_y : float, optional
        The relative vertical position of the line, by default .25
    **line_kwargs : dict
        Keywords passed to the line plotting function
    """
    if ax is None:
        ax = plt.gca()
    if axis == 1:
        diff = df - df.max(axis=1)[:, np.newaxis]
    else: 
        diff = df - df.max(axis=0)
    is_max = diff.abs() < tol
    maxima = np.argwhere(is_max.values).tolist()
    text_elements = [child for child in ax.get_children() 
                     if isinstance(child, Text)]
    for text in text_elements:
        x, y = text.get_position()
        pos = [int(y-0.5), int(x-0.5)]
        if pos in maxima and text.get_text() != '':
            text.set_fontweight('bold')
            line_opts = dict(lw=1, linestyle='-', color=text.get_color())
            line_opts.update(line_kwargs)
            plt.plot([x - line_length/2, x + line_length/2], 
                     [y + line_y, y + line_y],
                     **line_opts)
            

class _MyHeatMapper(_HeatMapper):
    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        mesh.update_scalarmappable()
        height, width = self.annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                       mesh.get_array(), mesh.get_facecolors(),
                                       self.annot_data.flat):
            if m is not np.ma.masked:
                lum = relative_luminance(color)
                text_color = ".15" if lum > .408 else "w"
                #------
                #
                # This is only change...
                annotation = self.fmt.format(val)
                # annotation = ("{:" + self.fmt + "}").format(val)
                #
                #------
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kws)
                ax.text(x, y, annotation, **text_kwargs)
                
def _heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g", annot_kws=None,
            linewidths=0, linecolor="white",
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=False, xticklabels="auto", yticklabels="auto",
            mask=None, ax=None, **kwargs):
    # Initialize the plotter object
    plotter = _MyHeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                          annot_kws, cbar, cbar_kws, xticklabels,
                          yticklabels, mask)

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax

def score_heatmap(means, stds=None, fmt=True, **kwargs):
    """Show scores as a heatmap, annotated by the mean values and
    the standard deviations (if passed). You can pass a format string
    using the mu and sigma variables for the mean and standard deviation
    respectively. If stds=None, fmt should be a string with one 
    (unnamed) variable. Note that means and stds should be pandas
    dataframes.
    """
    if stds is not None:
        assert means.shape == stds.shape
        if fmt == True: fmt = '${mu:.1f}^{{\pm {sigma:.1f}}}$'
        column_annotator = (
            lambda means, stds:
                [fmt.format(mu=mu, sigma=sigma) 
                 for mu, sigma in zip(means, stds)]
        )
        annotations = means.combine(stds, column_annotator)
        fmt = '{}'
    else:
        annotations=True
        if fmt == True: fmt = '{:.0f}'
    
    return _heatmap(means, annot=annotations, fmt=fmt, **kwargs)