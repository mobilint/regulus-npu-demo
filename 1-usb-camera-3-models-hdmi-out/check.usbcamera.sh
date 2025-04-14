#!/bin/bash
videos=`ls /dev/video*`
echo ${videos}
for i in ${videos}
do
	mjpg=`v4l2-ctl -d ${i} --all | grep usb`
	if [ ${#mjpg} != 0 ]; then
		yuyv=`v4l2-ctl -d ${i} --list-formats-ext | grep YUYV`
		if [ ${#yuyv} != 0 ]; then
			echo "usb camera device : ${i}"
		fi
	fi
done
