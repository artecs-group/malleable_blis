


for max_t in 2
do

	for t in 1 #2 #1 2 3 4
	do
		#export BLIS_JC_NT=${t} # loop 5 - n (externo)
		#loop 4 not enabled - k
		export BLIS_IC_NT=${t} # loop 3 - m
		#export BLIS_JR_NT=${t} # loop 2 - n
		#export BLIS_IR_NT=${t} # loop 1 - m (interno)

		echo $t ${max_t}

		./test_gemm.x $t ${max_t}  #> salida_${i}_threads.txt

	done
done
