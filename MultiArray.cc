/*******************************************************************************
 * Copyright (c) 2020 CEA
 * This program and the accompanying materials are made available under the 
 * terms of the Eclipse Public License 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0.
 *
 * SPDX-License-Identifier: EPL-2.0
 * Contributors: see AUTHORS file
 *******************************************************************************/

#include "MultiArray.h"
#include <CL/sycl.hpp>
#include <vector>

//#ifdef TEST

// ****************************************************************************
void dummy() {
  MultiArray<double, std::allocator<double>, 2, 3> a{0, 1, 3,
                             4, 5, 6};

  std::cout << "MultiArray test :" << std::endl;
  std::cout << "a : rank = " << a.dimension() << ", extent = " << a.size() << std::endl;
  std::cout << "a[0] : rank = " << a[0].dimension() << ", extent = " << a[0].size() << std::endl;
  std::cout << a << std::endl;

  size_t nbCells(3);
  size_t nbNodesOfCell(4);
  size_t five(5);
  
  MultiArray<int, std::allocator<int>, 0> b(five);
  std::cout << "b : rank = " << b.dimension() << ", extent = " << b.size() << std::endl;
  std::cout << b << std::endl << std::endl;

  MultiArray<float, std::allocator<float>, 0, 0, 2, 2> Ajr(nbCells, five, 2, 2);
  Ajr.initSize(nbCells, nbNodesOfCell, 2, 2);
  std::cout << "Ajr : rank = " << Ajr.dimension() << ", extent = " << Ajr.size() << std::endl;
  std::cout << Ajr << std::endl;

  cl::sycl::queue q;
  cl::sycl::usm_allocator<int, cl::sycl::usm::alloc::shared> my_alloc(q);
  constexpr size_t kSIZE(3);
  //std::vector<int, decltype(my_alloc)> my_toto(size, my_alloc);
  MultiArray<int, decltype(my_alloc), kSIZE, kSIZE> my_toto(my_alloc);
  auto toto(my_toto.data());
  q.submit([&](cl::sycl::handler &h) {
      h.parallel_for(cl::sycl::range<2>(kSIZE, kSIZE), [=](cl::sycl::item<2> idx) {
		    toto[idx.get_linear_id()] = idx.get_linear_id();
  	  });
    }).wait();
  std::cout << "Toto:" << std::endl << my_toto << std::endl;

  MultiArray<int, std::allocator<int>, 0, 1, 1> fail(five, 2, 2);
}
// *****************************************************************************

int main() {
  dummy();
}

//#endif