/* adept_openmp.cpp -- OpenMP-enabled Jacobian calculation for Adept library

    Copyright (C) 2013-2015 The University of Reading

    Author: Robin Hogan <r.j.hogan@reading.ac.uk>

    This file is part of the Adept library.


   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


   This file can and should be compiled even if OpenMP is not enabled.

*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include "adept.h"

namespace adept {

  // Compute the Jacobian matrix, parallelized using OpenMP. Normally
  // the user would call the jacobian or jacobian_forward functions,
  // and the OpenMP version would only be called if OpenMP is
  // available and the Jacobian matrix is large enough for
  // parallelization to be worthwhile.  Note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. The independents
  // and dependents must have already been identified with the
  // functions "independent" and "dependent", otherwise this function
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. In the
  // resulting matrix, the "m" dimension of the matrix varies
  // fastest. This is implemented using a forward pass, appropriate
  // for m>=n.
  void
  Stack::jacobian_forward_openmp(Real* jacobian_out)
  {
    if (independent_offset_.empty() || dependent_offset_.empty()) {
      throw(dependents_or_independents_not_identified());
    }

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than ADEPT_MULTIPASS_SIZE variables
    int n_block = (n_independent() + ADEPT_MULTIPASS_SIZE - 1)
      / ADEPT_MULTIPASS_SIZE;
    Offset n_extra = n_independent() % ADEPT_MULTIPASS_SIZE;
    
    int iblock;
    
#pragma omp parallel
    {
      std::vector<Block<ADEPT_MULTIPASS_SIZE,Real> > 
	gradient_multipass_b(max_gradient_);
      
#pragma omp for
      for (iblock = 0; iblock < n_block; iblock++) {
	// Set the offset to the dependent variables for this block
	Offset i_independent =  ADEPT_MULTIPASS_SIZE * iblock;
	
	Offset block_size = ADEPT_MULTIPASS_SIZE;
	// If this is the last iteration and the number of extra
	// elements is non-zero, then set the block size to the number
	// of extra elements. If the number of extra elements is zero,
	// then the number of independent variables is exactly divisible
	// by ADEPT_MULTIPASS_SIZE, so the last iteration will be the
	// same as all the rest.
	if (iblock == n_block-1 && n_extra > 0) {
	  block_size = n_extra;
	}
	
	// Set the initial gradients all to zero
	for (Offset i = 0; i < gradient_multipass_b.size(); i++) {
	  gradient_multipass_b[i].zero();
	}
	// Each seed vector has one non-zero entry of 1.0
	for (Offset i = 0; i < block_size; i++) {
	  gradient_multipass_b[independent_offset_[i_independent+i]][i] = 1.0;
	}
	// Loop forward through the derivative statements
	for (Offset ist = 1; ist < n_statements_; ist++) {
	  const Statement& statement = statement_[ist];
	  // We copy the LHS to "a" in case it appears on the RHS in any
	  // of the following statements
	  Block<ADEPT_MULTIPASS_SIZE,Real> a; // Initialized to zero
					      // automatically

	  // Loop through operations
	  for (Offset iop = statement_[ist-1].end_plus_one;
	       iop < statement.end_plus_one; iop++) {
	    // Loop through columns within this block; we hope the
	    // compiler can optimize this loop. Note that it is faster
	    // to always use ADEPT_MULTIPASS_SIZE, always known at
	    // compile time, than to use block_size, which is not, even
	    // though in the last iteration this may involve redundant
	    // computations.
	    if (multiplier_[iop] == 1.0) {
	      for (Offset i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
		//	      for (Offset i = 0; i < block_size; i++) {
		a[i] += gradient_multipass_b[offset_[iop]][i];
	      }
	    }
	    else {
	      for (Offset i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
		//	      for (Offset i = 0; i < block_size; i++) {
		a[i] += multiplier_[iop]*gradient_multipass_b[offset_[iop]][i];
	      }
	    }
	  }
	  // Copy the results
	  for (Offset i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	    gradient_multipass_b[statement.offset][i] = a[i];
	  }
	} // End of loop over statements
	// Copy the gradients corresponding to the dependent variables
	// into the Jacobian matrix
	for (Offset idep = 0; idep < n_dependent(); idep++) {
	  for (Offset i = 0; i < block_size; i++) {
	    jacobian_out[(i_independent+i)*n_dependent()+idep]
	      = gradient_multipass_b[dependent_offset_[idep]][i];
	  }
	}
      } // End of loop over blocks
    } // End of parallel section
  } // End of jacobian function



  // Compute the Jacobian matrix, parallelized using OpenMP.  Normally
  // the user would call the jacobian or jacobian_reverse functions,
  // and the OpenMP version would only be called if OpenMP is
  // available and the Jacobian matrix is large enough for
  // parallelization to be worthwhile.  Note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. The independents
  // and dependents must have already been identified with the
  // functions "independent" and "dependent", otherwise this function
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. In the
  // resulting matrix, the "m" dimension of the matrix varies
  // fastest. This is implemented using a reverse pass, appropriate
  // for m<n.
  void
  Stack::jacobian_reverse_openmp(Real* jacobian_out)
  {
    if (independent_offset_.empty() || dependent_offset_.empty()) {
      throw(dependents_or_independents_not_identified());
    }

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than ADEPT_MULTIPASS_SIZE variables
    int n_block = (n_dependent() + ADEPT_MULTIPASS_SIZE - 1)
      / ADEPT_MULTIPASS_SIZE;
    Offset n_extra = n_dependent() % ADEPT_MULTIPASS_SIZE;
    
    int iblock;

    // Inside the OpenMP loop, the "this" pointer may be NULL if the
    // adept::Stack pointer is declared as thread-local and if the
    // OpenMP memory model uses thread-local storage for private
    // data. If this is the case then local pointers to or copies of
    // the following members of the adept::Stack object may need to be
    // made: dependent_offset_ n_statements_ statement_ multiplier_
    // offset_ independent_offset_ n_dependent() n_independent().
    // Limited testing implies this is OK though.

#pragma omp parallel
    {
      std::vector<Block<ADEPT_MULTIPASS_SIZE,Real> > 
	gradient_multipass_b(max_gradient_);
      
#pragma omp for
      for (iblock = 0; iblock < n_block; iblock++) {
	// Set the offset to the dependent variables for this block
	Offset i_dependent =  ADEPT_MULTIPASS_SIZE * iblock;
	
	Offset block_size = ADEPT_MULTIPASS_SIZE;
	// If this is the last iteration and the number of extra
	// elements is non-zero, then set the block size to the number
	// of extra elements. If the number of extra elements is zero,
	// then the number of independent variables is exactly divisible
	// by ADEPT_MULTIPASS_SIZE, so the last iteration will be the
	// same as all the rest.
	if (iblock == n_block-1 && n_extra > 0) {
	  block_size = n_extra;
	}

	// Set the initial gradients all to zero
	for (Offset i = 0; i < gradient_multipass_b.size(); i++) {
	  gradient_multipass_b[i].zero();
	}
	// Each seed vector has one non-zero entry of 1.0
	for (Offset i = 0; i < block_size; i++) {
	  gradient_multipass_b[dependent_offset_[i_dependent+i]][i] = 1.0;
	}

	// Loop backward through the derivative statements
	for (Offset ist = n_statements_-1; ist > 0; ist--) {
	  const Statement& statement = statement_[ist];
	  // We copy the RHS to "a" in case it appears on the LHS in any
	  // of the following statements
	  Real a[ADEPT_MULTIPASS_SIZE];
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	  // For large blocks, we only process the ones where a[i] is
	  // non-zero
	  Offset i_non_zero[ADEPT_MULTIPASS_SIZE];
#endif
	  Offset n_non_zero = 0;
	  for (Offset i = 0; i < block_size; i++) {
	    a[i] = gradient_multipass_b[statement.offset][i];
	    gradient_multipass_b[statement.offset][i] = 0.0;
	    if (a[i] != 0.0) {
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	      i_non_zero[n_non_zero++] = i;
#else
	      n_non_zero = 1;
#endif
	    }
	  }

	  // Only do anything for this statement if any of the a values
	  // are non-zero
	  if (n_non_zero) {
	    // Loop through the operations
	    for (Offset iop = statement_[ist-1].end_plus_one;
		 iop < statement.end_plus_one; iop++) {
	      // Try to minimize pointer dereferencing by making local
	      // copies
	      register Real multiplier = multiplier_[iop];
	      register Real* __restrict gradient_multipass 
		= &(gradient_multipass_b[offset_[iop]][0]);
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	      // For large blocks, loop over only the indices
	      // corresponding to non-zero a
	      for (Offset i = 0; i < n_non_zero; i++) {
		gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	      }
#else
	      // For small blocks, do all indices
	      for (Offset i = 0; i < block_size; i++) {
	      //	      for (Offset i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
		gradient_multipass[i] += multiplier*a[i];
	      }
#endif
	    }
	  }
	} // End of loop over statement
	// Copy the gradients corresponding to the independent
	// variables into the Jacobian matrix
	for (Offset iindep = 0; iindep < n_independent(); iindep++) {
	  for (Offset i = 0; i < block_size; i++) {
	    jacobian_out[iindep*n_dependent()+i_dependent+i] 
	      = gradient_multipass_b[independent_offset_[iindep]][i];
	  }
	}
      } // End of loop over blocks
    } // end #pragma omp parallel
  } // end jacobian_reverse_openmp
} // End of namespace adept
