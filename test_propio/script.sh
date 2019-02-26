


for i in 2 #1 2 3 4
do
	#export BLIS_IC_NT=${i}

	#echo $i

	./test_gemm.x $i #> salida_${i}_threads.txt

done
