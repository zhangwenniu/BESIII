# BESIII<br>
Scripts for data analysis in the project BESIII track simulation. <br>

数据结构如下：<br>
--display.py<br>
--input<br>
&nbsp;|--MdcWirePosition_191003.csv<br>
&nbsp;|--digiMc<br>
&nbsp;&nbsp;&nbsp;|--digiMc_0_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_1_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_2_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_3_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_4_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_5_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_6_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_7_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_8_0.csv<br>
&nbsp;&nbsp;&nbsp;|--digiMc_9_0.csv<br>
输入数据分别对应了三组预处理文件，将会由程序自动生成到./input/fillzero,./input/premerge,./input/pretracked三个文件夹下，分别对应初始数据填充0，为每个点添加坐标信息，为每个坐标生成新的径迹值的数据。<br>
对应每张图像的pdf文件将会输出到./output/output文件夹下<br>
--output<br>
&nbsp;|--output<br>
同时，可以指定两种方式生成圆弧径迹路线，分别是linear和scatter。<br>
linear是用画线法绘制的，没有圆弧径迹的限制，是一条完整的弧线。<br>
scatter是用散点图方法绘制的，限制只在哪些有点出现的弧上进行绘制。<br>
使用方式：<br>
python display.py --method scatter --n 1000<br>
--method的可选参数是scatter和line，--n的参数对应需要打印多少组实验图像。