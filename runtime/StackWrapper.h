/*
 * StackWrapper.h
 *
 *  Created on: Jul 5, 2018
 *      Author: ahueck
 */

#ifndef RUNTIME_STACKWRAPPER_H_
#define RUNTIME_STACKWRAPPER_H_

#include <utility>

namespace typeart {

template <typename Container>
class StackWrapper {
 public:
  using value_type = typename Container::value_type;
  using size_type = typename Container::size_type;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  StackWrapper() = default;

  explicit StackWrapper(const Container& other) : c(other) {
  }

  explicit StackWrapper(Container&& other) : c(std::move(other)) {
  }

  inline Container& container() {
    return c;
  }

  inline const Container& container() const {
    return c;
  }

  inline void push_back(const value_type& a) {
    if (c.size() == index) {
      c.push_back(a);
    } else {
      c[index] = a;
    }
    ++index;
  }

  inline void free(size_type count) {
    index -= count;
  }

  inline size_type size() const {
    return index;
  }

  inline iterator begin() const {
    return c.begin();
  }

  inline iterator end() const {
    return c.begin() + index;
  }

  inline const_iterator cbegin() const {
    return c.cbegin();
  }

  inline const_iterator cend() const {
    return c.cbegin() + index;
  }

 private:
  Container c;
  size_type index{0};
};

}  // namespace typeart

#endif /* RUNTIME_STACKWRAPPER_H_ */
