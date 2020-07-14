//
// Created by ahueck on 14.07.20.
//

#ifndef TYPEART_DATAIO_H
#define TYPEART_DATAIO_H

#include <memory>
#include <string>

namespace typeart {

class DataDB;

class DataIO {
 private:
  DataDB& dataDB;

 public:
  explicit DataIO(DataDB& config);
  bool load(const std::string& file);
  bool store(const std::string& file) const;
};

}  // namespace typeart
#endif  // TYPEART_DATAIO_H
