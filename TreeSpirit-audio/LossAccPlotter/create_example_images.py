"""This script creates some example plots for the README.md."""
from __future__ import print_function, division
from laplotter import LossAccPlotter
from check_laplotter import show_chart, add_noise
import numpy as np

def main():
    """Create the example plots in the following way:
    1. Generate example data (all plots use more or less the same data)
    2. Generate plot 1: "standard" example with loss and accuracy
    3. Generate plot 2: Same as 1, but only loss / no accuracy
    4. Generate plot 3: Same as 1, but no validation lines (only training dataset)
    5. Generate plot 4: Same as 1, but use only every 10th validation datapoint
                        (i.e. resembles real world scenario where you rarely
                        validate your machine learning method)
    """
    plotter = LossAccPlotter()
    nb_points = 500

    # loss_train = add_noise(np.linspace(0.9, 0.1, num=nb_points), 0.025)
    # loss_val = add_noise(np.linspace(0.7, 0.3, num=nb_points), 0.045)
    # acc_train = add_noise(np.linspace(0.52, 0.95, num=nb_points), 0.025)
    # acc_val = add_noise(np.linspace(0.65, 0.75, num=nb_points), 0.045)





    loss_train = np.array([2.5777 ,
1.9580,
1.9528 ,
1.8874 ,
1.9383 ,
1.8387,
1.8481 ,
1.8171 ,
1.3788 ,
1.5282 ,
1.5747 ,
1.5794 ,
1.4200 ,
1.4265 ,
1.3506 ,
1.3367 ,
1.4058 ,
1.3526 ,
1.3275 ,
1.2919 ,
1.2936 ,
1.3511 ,
1.2991 ,
1.2887 ,
1.3701 ,
1.2567 ,
1.2362 ,
1.2759 ,
1.1197 ,
1.1445 ,
1.1733 ,
1.1645 ,
1.2224 ,
1.2326 ,
1.2178 ,
1.2115 ,
1.1643 ,
1.1258 ,
1.1118 ,
1.1239 ,
1.1689 ,
1.1675 ,
1.1510 ,
1.1720 ,
1.1511 ,
1.1322 ,
1.1272 ,
1.1191 ,
1.1408 ,
1.1094 ,
1.1322 ,
1.1003 ,
1.0952 ,
1.0744 ,
1.1087,
1.0657 ,
1.0698,
1.0775 ,
1.0660 ,
1.0802 ,
1.0422 ,
1.0478 ,
1.1066 ,
1.0821 ,
1.0572 ,
1.0630 ,
1.1096 ,
1.0872 ,
1.0725 ,
1.0526 ,
1.0674 ,
1.0413 ,
1.0390 ,
1.0325 ,
1.0328 ,
1.0749 ,
1.0734 ,
1.0195 ,
1.0348 ,
1.0220 ,
1.0419 ,
1.0166 ,
1.0196 ,
1.0196 ,
1.0268,
1.0487 ,
1.0516,
1.0738 ,
1.0590 ,
1.0531 ,
1.0529 ,
1.0877 ,
1.0604 ,
1.0394 ,
1.0348 ,
1.0574 ,
1.0325 ,
1.0332 ,
1.0376 ,
1.0380 ,
1.0366 ,
1.0330 ,
1.0325 ,
1.0213 ,
1.0185 ,
1.0364 ,
1.0327 ,
1.0351 ,
1.0364 ,
1.0393 ,
1.0297 ,
1.0583,
1.0429 ,
1.0311 ,
1.0373 ,
1.0283 ,
1.0324 ,
1.0178 ,
1.0204 ,
1.0108 ,
1.0131 ,
1.0134 ,
1.0160 ,
1.0063 ,
1.0223 ,
1.0229 ,
1.0203 ,
1.0181 ,
1.0411,
1.0328 ,
1.0316 ,
1.0208 ,
1.0182 ,
1.0200 ,
1.0164 ,
1.0105 ,
1.0138,
1.0277 ,
1.0267 ,
1.0470 ,
1.0264 ,
1.0193 ,
1.0164,
1.0035 ,
1.0168 ,
1.0227 ,
1.0294 ,
1.0230 ,
1.0153 ,
1.0071 ,
1.0013 ,
1.0165 ,
1.0220 ,
1.0148 ,
1.0533 ,
1.0302 ,
1.0315,
1.0200,
1.0070 ,
1.0252 ,
1.0226 ,
1.0227,
1.0134 ,
1.0277 ,
1.0139 ,
1.0125 ,
1.0130 ,
1.0042 ,
1.0049 ,
0.9981 ,
1.0082 ,
1.0019 ,
1.0089,
1.0092 ,
1.0070 ,
0.9989 ,
0.9962 ,
1.0073 ,
1.0120 ,
1.0040 ,
1.0047 ,
1.0064 ,
1.0096 ,
1.0008 ,
1.0010 ,
1.0159 ,
1.0210 ,
1.0160 ,
1.0152 ,
0.9899 ,
1.0049 ,
1.0025,
0.9977 ,
0.9982 ,
1.0050 ,
1.0281,
1.0227 ,
1.0182 ,
0.9902 ,
1.0006 ,
0.9962,
0.9947 ,
1.0155,
1.0118 ,
1.0128 ,
1.0081 ,
0.9987 ,
1.0003 ,
1.0044 ,
1.0159 ,
0.9860 ,
1.0024 ,
1.0020 ,
1.0006 ,
1.0194 ,
1.0170 ,
1.0097 ,
1.0067 ,
1.0112 ,
1.0007 ,
1.0002 ,
0.9984 ,
0.9939 ,
0.9944 ,
1.1987 ,
1.0014 ,
1.0020 ,
0.9999 ,
1.0009 ,
1.0005 ,
0.9895 ,
1.0036 ,
1.0023 ,
0.9995 ,
0.9991 ,
0.9900 ,
0.9861 ,
0.9882 ,
0.9929 ,
1.0109 ,
1.0041 ,
1.0007 ,
0.9973 ,
1.0020 ,
0.9980 ,
0.9975 ,
1.0268 ,
1.0271 ,
1.0134 ,
1.0101 ,
1.0066 ,
0.9956 ,
0.9895 ,
0.9901 ,
0.9744,
0.9902 ,
0.9860 ,
0.9851 ,
1.0012 ,
1.0021,
0.9934 ,
0.9966 ,
0.9808 ,
0.9953 ,
0.9933 ,
0.9895 ,
0.9839 ,
0.9922 ,
0.9938 ,
0.9925 ,
1.0103 ,
0.9970 ,
0.9968 ,
0.9932,
0.9900 ,
0.9844 ,
0.9903 ,
0.9868 ,
0.9774 ,
0.9774 ,
0.9835 ,
0.9886 ,
1.0003 ,
0.9982 ,
0.9944,
0.9924 ,
0.9866 ,
0.9921 ,
0.9913 ,
0.9906,
1.0129 ,
1.0058,
0.9987 ,
0.9955 ,
0.9824 ,
0.9904 ,
0.9906 ,
0.9937 ,
0.9705 ,
0.9793 ,
0.9879 ,
0.9878 ,
0.9794 ,
0.9907 ,
0.9923 ,
0.9932
 ]);

    acc_train = np.array([ 0.3800
    , 0.6600
    , 0.7400
    , 0.7850
    , 0.9100
    , 0.9200
    , 0.9133
    , 0.9125
    , 0.9400
    , 0.9000
    , 0.8433
    , 0.8100
    , 0.8200
    , 0.8650
    , 0.8800
    , 0.8875
    , 0.9100
    , 0.9200
    , 0.9067
    , 0.9175
    , 0.9200
    , 0.9050
    , 0.9200
    , 0.9250
    , 0.9400
    , 0.9500
    , 0.9367
    , 0.9325
    , 0.9600
    , 0.9550
    , 0.9500
    , 0.9525
    , 0.9200
    , 0.9250
    , 0.9333
    , 0.9400
    , 0.9400
    , 0.9450
    , 0.9467
    , 0.9475
    , 0.9500
    , 0.9450
    , 0.9500
    , 0.9425
    , 0.9467
    , 0.9475
    , 0.9400
    , 0.9450
    , 0.9550
    , 0.9575
    , 0.9600
    , 0.9600
    , 0.9500
    , 0.9650
    , 0.9625
    , 0.9775
    , 0.9667
    , 0.9650
    , 0.9667
    , 0.9625
    , 0.9750
    , 0.9750
    , 0.9550
    , 0.9650
    , 0.9700
    , 0.9700
    , 0.9600
    , 0.9600
    , 0.9625
    , 0.9725
    , 0.9625
    , 0.9733
    , 0.9750
    , 0.9800
    , 0.9775
    , 0.9700
    , 0.9675
    , 0.9800
    , 0.9725
    , 0.9800
    , 0.9700
    , 0.9800
    , 0.9700
    , 0.9733
    , 0.9700
    , 0.9800
    , 0.9725
    , 0.9700
    , 0.9700
    , 0.9700
    , 0.9700
    ,0.9600
    ,0.9700
    , 0.9733
    , 0.9750
    , 0.9700
    , 0.9800
    , 0.9800
    , 0.9750
    , 0.9700
    , 0.9700
    , 0.9733
    , 0.9750
    , 0.9800
    , 0.9800
   , 0.9700
    , 0.9725
    , 0.9700
    ,0.9675
    , 0.9700
    , 0.9775
    ,0.9600
    , 0.9650
    ,0.9667
    , 0.9625
    , 0.9800
   , 0.9800
    , 0.9800
   , 0.9750
    , 0.9900
    , 0.9850
    , 0.9833
    , 0.9825
    , 0.9900
    , 0.9800
    , 0.9767
    , 0.9775
    , 0.9800
    , 0.9600
    , 0.9633
    , 0.9650
    , 0.9800
   , 0.9775
    , 0.9800
    , 0.9825
    , 0.9800
    , 0.9775
    , 0.9800
    , 0.9750
    , 0.9600
    , 0.9775
    , 0.9733
    , 0.9750
    , 0.9800
    , 0.9775
    , 0.9733
    , 0.9725
    , 0.9700
    , 0.9767
    , 0.9825
    , 0.9800
    , 0.9750
   , 0.9700
    , 0.9725
    , 0.9500
    , 0.9650
    , 0.9667
    , 0.9725
   , 0.9800
    , 0.9650
    , 0.9633
   , 0.9675
    , 0.9825
    , 0.9700
    , 0.9850
    , 0.9800
    , 0.9775
    , 0.9800
    , 0.9825
    , 0.9900
    ,0.9825
    , 0.9800
    , 0.9775
    , 0.9833
    , 0.9825
    , 0.9800
    ,0.9875
    , 0.9767
   , 0.9725
    , 0.9800
    , 0.9800
    , 0.9900
    , 0.9775
    , 0.9850
    , 0.9850
    , 0.9800
    , 0.9750
    , 0.9767
    , 0.9775
   , 0.9800
    , 0.9775
    , 0.9750
    , 0.9800
    , 0.9775
    , 0.9700
    , 0.9650
    , 0.9667
   , 0.9750
    , 0.9800
    , 0.9750
    , 0.9800
    , 0.9825
    , 0.9700
    , 0.9700
    , 0.9733
    , 0.9750
    , 0.9800
    , 0.9800
    , 0.9833
    , 0.9775
    , 0.9900
    , 0.9750
    , 0.9767
    , 0.9750
    , 0.9600
    , 0.9650
   , 0.9700
    , 0.9725
    , 0.9700
    , 0.9750
    , 0.9733
    , 0.9725
    , 0.9800
    , 0.9850
    , 0.9800
    , 0.9775
    , 0.9900
    , 0.9850
    , 0.9833
    , 0.9825
    , 0.9900
    , 0.9850
    , 0.9800
   , 0.9800
    , 0.9800
    , 0.9900
    , 0.9900
    ,0.9900
    , 0.9900
    , 0.9750
    , 0.9800
    , 0.9800
    , 0.9800
    , 0.9750
    ,0.9767
    , 0.9775
    , 0.9700
   , 0.9600
    , 0.9667
    , 0.9675
    ,0.9700
    , 0.9800
    , 0.9867
   , 0.9850
    , 0.9900
    , 0.9800
   , 0.9867
    , 0.9875
    , 0.9700
    ,0.9700
    , 0.9767
    , 0.9775
   ,0.9900
    , 0.9800
    , 0.9767
    , 0.9800
    , 0.9900
    , 0.9800
    , 0.9800
    , 0.9825
    , 0.9700
    , 0.9800
    , 0.9833
    , 0.9850
    , 0.9900
    , 0.9900
    , 0.9833
    , 0.9850
    , 0.9900
    , 0.9900
    , 0.9867
    , 0.9825
    , 0.9800
    , 0.9800
    , 0.9833
    ,0.9825
    , 1.0000
    , 0.9900
    , 0.9867
    , 0.9875
    , 0.9800
    , 0.9800
    , 0.9833
    , 0.9850
    , 0.9800
    , 0.9800
    , 0.9767
    , 0.9750
    , 0.9900
    , 0.9850
    , 0.9833
    ,0.9850
    , 0.9900
    , 0.9800
   , 0.9800
    , 0.9775
    ]);

    # Normal example plot
    lap = LossAccPlotter(save_to_filepath="example_plot.png")
    show_chart(loss_train, np.array([]), acc_train,  np.array([]), lap=lap,
               title="Plot with Loss and Accuracy")


# As the plot is non-blocking, we should call plotter.block() at the end, to
# change it to the blocking-mode. Otherwise the program would instantly end
# and thereby close the plot.


    # Plot showing only the results of the loss function (accuracy off)
    # lap = LossAccPlotter(show_acc_plot=False,
    #                      save_to_filepath="example_plot_loss.png")
    # show_chart(loss_train, loss_val, acc_train, acc_val, lap=lap,
    #            title="Example Plot, only Loss Function")
    #
    # # Plot showing only training dataset values (but for both loss and accuracy)
    # lap = LossAccPlotter(save_to_filepath="example_plot_only_training.png")
    # show_chart(loss_train, np.array([]), acc_train, np.array([]), lap=lap,
    #            title="Example Plot, only Training Dataset / no Validation Dataset")
    #
    # # Plot with a different update interval for training and validation dataset
    # # (i.e. only one validation value for every 10 training values)
    # #
    # # Set 9 out of 10 validation values to -1, which will be transformed into
    # # None in show_chart(). (same technique as in check_laplotter.py)
    # nb_points_train = nb_points
    # nb_points_val = int(nb_points * 0.1)
    # all_indices = np.arange(0, nb_points_train-1, 1)
    # keep_indices = np.arange(0, nb_points_train-1, int(nb_points_train / nb_points_val))
    # set_to_none_indices = np.delete(all_indices, keep_indices)
    # loss_val[set_to_none_indices] = -1.0
    # acc_val[set_to_none_indices] = -1.0
    # lap = LossAccPlotter(show_acc_plot=False,
    #                      save_to_filepath="example_plot_update_intervals.png")
    # show_chart(loss_train, loss_val, acc_train, acc_val, lap=lap,
    #            title="Example Plot with different Update Intervals for Training " \
    #                  "and Validation Datasets")

if __name__ == "__main__":
    main()
