#!/bin/bash
if [ "$1" != "" ]; then
	dlib_root=$1
        install_dir=$1/../local
	if [ ! -f $install_dir/lib/libdlib.a ]; then
		mkdir -p $install_dir
		mkdir -p $dlib_root/build
		cd $dlib_root/build && cmake .. -DCMAKE_INSTALL_PREFIX=$install_dir -DDLIB_NO_GUI_SUPPORT=1 && cmake --build . --config Release --target dlib && make install
	else
		echo "dlib already present, skipping"
	fi
else
	echo use $0 dlib_root_dir
fi
