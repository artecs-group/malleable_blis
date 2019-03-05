#include "blis.h"




void libblis_test_ceil_pow2( obj_t* alpha )
{
	double alpha_r;
	double alpha_i;

	bli_getsc( alpha, &alpha_r, &alpha_i );

	alpha_r = pow( 2.0, ceil( log2( alpha_r ) ) );

	bli_setsc( alpha_r, alpha_i, alpha );
}

void libblis_test_vobj_randomize( bool_t normalize, obj_t* x )
{
	//if ( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
		bli_randv( x );
	//else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
	//	bli_randnv( x );

	if ( normalize )
	{
		num_t dt   = bli_obj_dt( x );
		num_t dt_r = bli_obj_dt_proj_to_real( x );
		obj_t kappa;
		obj_t kappa_r;

		bli_obj_scalar_init_detached( dt,   &kappa );
		bli_obj_scalar_init_detached( dt_r, &kappa_r );

		// Normalize vector elements. The following code ensures that we
		// always invert-scale by whole power of two.
		bli_normfv( x, &kappa_r );
		libblis_test_ceil_pow2( &kappa_r );
		bli_copysc( &kappa_r, &kappa );
		bli_invertsc( &kappa );
		bli_scalv( &kappa, x );
	}
}

void libblis_test_trsm_check
     (
//       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         b_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( b );
	num_t  dt_real = bli_obj_dt_proj_to_real( b );

	dim_t  m       = bli_obj_length( b );
	dim_t  n       = bli_obj_width( b );

	obj_t  norm;
	obj_t  t, v, w, z;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized and triangular.
	// - b_orig is randomized.
	// Note:
	// - alpha should have a non-zero imaginary component in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   B := alpha * inv(transa(A)) * B_orig    (side = left)
	//   B := alpha * B_orig * inv(transa(A))    (side = right)
	//
	// is functioning correctly if
	//
	//   normf( v - z )
	//
	// is negligible, where
	//
	//   v = B * t
	//
	//   z = ( alpha * inv(transa(A)) * B ) * t     (side = left)
	//     = alpha * inv(transa(A)) * B * t
	//     = alpha * inv(transa(A)) * w
	//
	//   z = ( alpha * B * inv(transa(A)) ) * t     (side = right)
	//     = alpha * B * tinv(ransa(A)) * t
	//     = alpha * B * w

	bli_obj_scalar_init_detached( dt_real, &norm );

	if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, m, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}
	else // else if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, n, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}

	//libblis_test_vobj_randomize( params, TRUE, &t );
	libblis_test_vobj_randomize( TRUE, &t );

	bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &v );

	if ( bli_is_left( side ) )
	{
		bli_gemv( alpha, b_orig, &t, &BLIS_ZERO, &w );
		bli_trsv( &BLIS_ONE, a, &w );
		bli_copyv( &w, &z );
	}
	else
	{
		bli_copyv( &t, &w );
		bli_trsv( &BLIS_ONE, a, &w );
		bli_gemv( alpha, b_orig, &w, &BLIS_ZERO, &z );
	}

	bli_subv( &z, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w );
	bli_obj_free( &z );
}


