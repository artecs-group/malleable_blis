#include "blis.h"

void libblis_test_trsm_check
     (
//       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         b_orig,
       double*        resid
      );
