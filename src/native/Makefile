all: pm_cpp_ext.cpython-36m-x86_64-linux-gnu.so #pm_cpp_ext.so


pm.o: pm.cpp pm/pm.h
	g++ -Wall -std=c++11 -fPIC pm.cpp -c

#pm.a: pm.o
#	ar rvs pm.a pm.o

libpm.so: pm.o
	g++ -shared pm.o -o libpm.so


pm_cpp_ext.cpython-36m-x86_64-linux-gnu.so: pm_cpp_ext.pyx setup_pm_cpp_ext.py libpm.so
#pm_cpp_ext.so: pm_cpp_ext.pyx setup_pm_cpp_ext.py libpm.so
	python3 setup_pm_cpp_ext.py build_ext --inplace

test: pm_cpp_ext.cpython-36m-x86_64-linux-gnu.so
	LD_LIBRARY_PATH=. ./test_pm_cpp_ext.py

clean:
	rm -rf *~ *.pyc build pm_cpp_ext.cpp pm.o

very_clean: clean
	rm -f pm_cpp_ext.so libpm.so pm_cpp_ext.cpython-36m-x86_64-linux-gnu.so
