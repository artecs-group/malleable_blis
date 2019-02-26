/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#define _DEFAULT_SOURCE

#include <unistd.h>
#include "blis.h"
#include "test_gemm_check.h"
#include <pthread.h>

//#define PRINT

#define USE_EXPERT
#define USE_CONTROLLER

#ifdef USE_CONTROLLER

// Define minimum and maximum time between thread changes.
#define MIN_TIME 1000000
#define MAX_TIME 4000000

// Define minimum and maximum number of threads.
#define MIN_THREADS 1
#define MAX_THREADS 4

#define TEST_2

struct arg_thread {

	cntx_t * context;
	rntm_t * rntm;
	int thread;
};

void * controller( void * info)
{
	struct arg_thread *args = info;
	cntx_t * cntx = (cntx_t*) args->context;
	rntm_t * rntm = (rntm_t*) args->rntm;

	int nth_l3 = 0;
	useconds_t sleep_useconds = 0;

#if 1
	while( 1 )
	{

		#ifdef TEST_1
			// Select a random number of threads for L3 loop.
			nth_l3 = rand() % (MAX_THREADS + 1 - MIN_THREADS) + MIN_THREADS;

			// Set active threads on loop 3.
			bli_rntm_set_active_ways( 1, 1, nth_l3, 1, 1, rntm );

			// Sleep a random time between MIN_TIME and MAX_TIME.
			sleep_useconds = rand() % (MAX_TIME + 1 - MIN_TIME) + MIN_TIME;
			usleep( sleep_useconds );

			printf( "Set number of active threads on L3 to %d and slept %d usecs\n", nth_l3, sleep_useconds );
		#else

			#ifdef TEST_2

				nth_l3++;
				if(nth_l3 > MAX_THREADS)
					nth_l3 = 1;

				bli_rntm_set_active_ways( 1, 1, nth_l3, 1, 1, rntm );
		
				sleep_useconds = MAX_TIME;

				printf( "Set number of active threads on L3 to %d and slept %d usecs\n", nth_l3, sleep_useconds );
		
				usleep( sleep_useconds );
			#endif
		#endif
	}
#endif

#ifdef TEST_3
	nth_l3=args->thread;
	printf( "Set number of active threads on L3 to %d\n", nth_l3 );
	sleep_useconds = 34000000;
	usleep(sleep_useconds);
	nth_l3 = MAX_THREADS - args->thread;
	bli_rntm_set_active_ways( 1, 1, nth_l3, 1, 1, rntm );
	printf( "Set number of active threads on L3 to %d\n", nth_l3 );
#endif

	return NULL;

}
#endif // USE_CONTROLLER

void libblis_test_ceil_pow2_local( obj_t* alpha )
{
	double alpha_r;
	double alpha_i;

	bli_getsc( alpha, &alpha_r, &alpha_i );

	alpha_r = pow( 2.0, ceil( log2( alpha_r ) ) );

	bli_setsc( alpha_r, alpha_i, alpha );
}


void libblis_test_mobj_randomize( bool_t normalize, obj_t* a )
{
	//if ( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
		bli_randm( a );
	//else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
	//	bli_randnm( a );

	if ( normalize )
	{
#if 0
		num_t dt      = bli_obj_dt( a );
		dim_t max_m_n = bli_obj_max_dim( a );
		obj_t kappa;

		bli_obj_scalar_init_detached( dt, &kappa );

		// Normalize vector elements by maximum matrix dimension.
		bli_setsc( 1.0/( double )max_m_n, 0.0, &kappa );
		bli_scalm( &kappa, a );
#endif
		num_t dt   = bli_obj_dt( a );
		num_t dt_r = bli_obj_dt_proj_to_real( a );
		obj_t kappa;
		obj_t kappa_r;

		bli_obj_scalar_init_detached( dt,   &kappa );
		bli_obj_scalar_init_detached( dt_r, &kappa_r );

		// Normalize matrix elements.
		bli_norm1m( a, &kappa_r );
		libblis_test_ceil_pow2_local( &kappa_r );
		bli_copysc( &kappa_r, &kappa );
		bli_invertsc( &kappa );
		bli_scalm( &kappa, a );
	}
}


int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input, k_input;
	num_t dt;
	int   r, n_repeats;
	trans_t  transa;
	trans_t  transb;
	f77_char f77_transa;
	f77_char f77_transb;

	double dtime;
	double dtime_save;
	double gflops;
	double resid=0.0;

	int hilos;
	if(argc != 2)
		hilos = 4;
	else
		hilos = atoi(argv[1]);

	cntx_t cntx;
	rntm_t rntm = BLIS_RNTM_INITIALIZER;

	bli_cntx_init_haswell( &cntx );
#ifdef USE_CONTROLLER
	bli_rntm_set_ways( 1, 1, MAX_THREADS, 1, 1, &rntm );
#else
	bli_rntm_set_ways( 1, 1, hilos, 1, 1, &rntm );
#endif

	bli_rntm_set_active_ways( 1, 1, hilos, 1, 1, &rntm );


	//bli_cntx_print( &cntx );

#ifdef USE_CONTROLLER
	int s;
	void * res;
	pthread_t controller_thread;
	struct arg_thread args;
	args.context = &cntx;
	args.rntm = &rntm;
	if(argc != 2)
		args.thread = MAX_THREADS;
	else
		args.thread = hilos;
	
	if( pthread_create( &controller_thread, NULL, controller, &args ) )
        {
		fprintf(stderr, "Error creating thread\n");
		return 1;
	}
#endif

	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 1;

#ifndef PRINT
	p_begin = 14000;
	//p_begin = 1000;
	p_end   = 17000;
	//p_end   = 10000;
	p_inc   = 1000;

	m_input = -1;
	n_input = -1;
	k_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 5;
	k_input = 6;
	n_input = 4;
#endif

#if 1
	dt = BLIS_FLOAT;
	//dt = BLIS_DOUBLE;
#else
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;
#endif

	transa = BLIS_NO_TRANSPOSE;
	transb = BLIS_NO_TRANSPOSE;

	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_trans( transb, &f77_transb );


	printf("\n***********************************************************************************\n\n");
	printf("Test for the evaluation of the GEMM routine using %d threads\n\n", hilos);
	printf("***********************************************************************************\n\n");
	printf("%27s - %6s %6s %6s\t%7s\t%7s\t%9s\n","","m","n","k"," Time  ","GFLOPS","RESID");


	for ( p = p_begin; p <= p_end; p += p_inc )
	{
		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		bli_obj_create( dt, m, k, 0, 0, &a );
		bli_obj_create( dt, k, n, 0, 0, &b );
		bli_obj_create( dt, m, n, 0, 0, &c );
		bli_obj_create( dt, m, n, 0, 0, &c_save );

		libblis_test_mobj_randomize(TRUE, &a );
		libblis_test_mobj_randomize(TRUE, &b );
		libblis_test_mobj_randomize(TRUE, &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transb, &b );

		bli_setsc( 1.2, 0.0, &alpha );
		bli_setsc( 0.9, 0.0, &beta );


		bli_copym( &c, &c_save );
	
		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );


			dtime = bli_clock();


#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "" );
			bli_printm( "b", &b, "%4.1f", "" );
			bli_printm( "c", &c, "%4.1f", "" );
#endif
		//printf("Start computation\n");

		if ( bli_is_float( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			float*   alphap = bli_obj_buffer( &alpha );
			float*   ap     = bli_obj_buffer( &a );
			float*   bp     = bli_obj_buffer( &b );
			float*   betap  = bli_obj_buffer( &beta );
			float*   cp     = bli_obj_buffer( &c );

#ifdef USE_EXPERT
			printf("Start Computation\n");
			bli_sgemm_ex( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
                                   mm, nn, kk, 
                                   alphap,
								   ap, 1, lda,
								   bp, 1, ldb,
                                   betap,
								   cp, 1, ldc,
								   &cntx, &rntm );
#else
			sgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
#endif
		}
		else if ( bli_is_double( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  bp     = bli_obj_buffer( &b );
			double*  betap  = bli_obj_buffer( &beta );
			double*  cp     = bli_obj_buffer( &c );

#ifdef USE_EXPERT
			bli_dgemm_ex( BLIS_NO_TRANSPOSE,
			           BLIS_NO_TRANSPOSE,
                                   mm, nn, kk, 
                                   alphap, ap, 1, lda,
                                           bp, 1, ldb,
                                   betap,  cp, 1, ldc, &cntx, &rntm );
#else
			dgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
#endif
		}
		else if ( bli_is_scomplex( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			scomplex*  alphap = bli_obj_buffer( &alpha );
			scomplex*  ap     = bli_obj_buffer( &a );
			scomplex*  bp     = bli_obj_buffer( &b );
			scomplex*  betap  = bli_obj_buffer( &beta );
			scomplex*  cp     = bli_obj_buffer( &c );

			cgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_dcomplex( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			dcomplex*  alphap = bli_obj_buffer( &alpha );
			dcomplex*  ap     = bli_obj_buffer( &a );
			dcomplex*  bp     = bli_obj_buffer( &b );
			dcomplex*  betap  = bli_obj_buffer( &beta );
			dcomplex*  cp     = bli_obj_buffer( &c );

			zgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

		libblis_test_gemm_check(&alpha, &a, &b, &beta, &c, &c_save, &resid );

#ifdef USE_EXPERT
		printf( "data_gemm_expert_%d", hilos );
#else
		printf( "data_gemm_blis" );
#endif
		printf( "( %2lu, 1:6 ) = [ %6lu %6lu %6lu\t%4.3f\t%4.3f\t%1.3e ];\n",
		        ( unsigned long )(p - p_begin + 1)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
		        ( unsigned long )n, dtime_save, gflops, resid );
		resid=0.0;

		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

#ifdef USE_CONTROLLER
	printf( "Cancelling controller...\n" );
	s = pthread_cancel( controller_thread );
	if ( s != 0 )
		printf( "pthread_cancel error\n" );

	/* Join with thread to see what its exit status was */

	s = pthread_join( controller_thread, &res );
	if ( s != 0 )
		printf( "pthread_join error\n" );

	if ( res == PTHREAD_CANCELED )
		printf( "Controller canceled\n" );
	else
		printf( "There was an error cancelling controller\n" );
#endif

	//bli_finalize();
	bli_cntx_clear( &cntx );


	return 0;
}

