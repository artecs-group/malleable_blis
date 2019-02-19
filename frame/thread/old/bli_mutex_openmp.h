/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2018, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#ifndef BLIS_MUTEX_OPENMP_H
#define BLIS_MUTEX_OPENMP_H

// Define mutex_t for situations when OpenMP multithreading is enabled.
#ifdef BLIS_ENABLE_OPENMP

#include <omp.h>

// Define mtx_t.

typedef struct mtx_s
{
	omp_lock_t mutex;
} mtx_t;

// Define functions to operate on OpenMP-based mtx_t.

static void bli_mutex_init( mtx_t* m )
{
	omp_init_lock( &(m->mutex) );
}

static void bli_mutex_finalize( mtx_t* m )
{
	omp_destroy_lock( &(m->mutex) );
}

static void bli_mutex_lock( mtx_t* m )
{
	omp_set_lock( &(m->mutex) );
}

static void bli_mutex_unlock( mtx_t* m )
{
	omp_unset_lock( &(m->mutex) );
}

#endif

#endif

