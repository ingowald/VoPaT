#include "vopat/common.h"
#include <netcdf>
#include <fstream>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;
using namespace vopat;


int main(int ac, char **av)
{
  std::string inFileName = av[1];
  std::string varName = "qi";
  std::string outFileBase = "/space/ncToRaw";

  NcFile dataFile(inFileName, NcFile::read);
  // std::multimap<std::string,NcVar> vars = dataFile.getVars();
  // for (auto _var : vars) {
  //   auto var = _var.second;
  //   std::cout << "found variable '" << _var.first << "' dims " << var.getDimCount() << std::endl;
  //   std::cout << "   dims ";
  //   for (auto d : var.getDims()) std::cout << " " << d.getSize();
  //   std::cout << std::endl;
  // }

  // std::multimap<std::string,NcDim> dims = dataFile.getDims();
  // for (auto dim : dims)
  //   std::cout << "found dims '" << dim.first << "' : " << dim.second.getSize() << std::endl;


  std::cout << "picking var '" << varName << "'" << std::endl;
  auto var = dataFile.getVar(varName);
  std::cout << "   dims ";
  for (auto d : var.getDims()) std::cout << " " << d.getSize();
  std::cout << std::endl;
  PRINT(var.getType().getName());

  vec3i dims(var.getDims()[1].getSize(),
             var.getDims()[2].getSize(),
             var.getDims()[3].getSize());
  PRINT(dims);
  
  vector<size_t> startp,countp;
  startp.push_back(0);
  startp.push_back(0);
  startp.push_back(0);
  startp.push_back(0);
  countp.push_back(1);
  countp.push_back(dims.x);
  countp.push_back(dims.y);
  countp.push_back(dims.z);

  std::vector<float> values(dims.x*size_t(dims.y)*dims.z);
  var.getVar(startp,countp,values.data());
  PING;
  int numFound = 0;
  const int numSearchMax = 10;
  for (int i=0;i<values.size() && numFound < numSearchMax;i++) {
    if (values[i] != 0.f) {
      std::cout << "found value " << values[i] << " at pos " << i << std::endl;
      numFound++;
    }
  }
  PRINT(numFound);PRINT(numSearchMax);
  std::string fileName = outFileBase + "_" + var.getName() + "_"
    + std::to_string(dims[0]) + "x"
    + std::to_string(dims[1]) + "x"
    + std::to_string(dims[2]) + "_float.raw";
  std::ofstream out(fileName,std::ios::binary);
  out.write((const char *)values.data(),values.size()*sizeof(values[0]));
  std::cout << "written to " << fileName << std::endl;
}
