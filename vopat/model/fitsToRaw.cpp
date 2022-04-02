#include "vopat/common.h"
#include <fitsio.h>
#include <fstream>
#include <stdlib.h>

using namespace vopat;


int main(int ac, char **av)
{
  std::string inFileName = av[1];
  std::string varName = "unknown";
  std::string outFileBase = "/fast/vopat/fits";

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-o")
      outFileBase = av[++i];
    else if (arg == "-v" || arg == "--variable")
      varName = av[++i];
    else throw std::runtime_error("unknown cmdline arg '"+arg+"'");
  }
  if (inFileName.empty())
    throw std::runtime_error("no input file name specified....");

  int status = 0;
  fitsfile *dataFile;
  fits_open_file(&dataFile,inFileName.c_str(),READONLY,&status);

  if (status != 0) {
    fits_report_error(stderr,status);
    fits_close_file(dataFile,&status);
    return EXIT_FAILURE;
  }

  // get number of "header-data-units"
  int numHDUs = 0;
  fits_get_num_hdus(dataFile,&numHDUs,&status);
  if (numHDUs == 0) {
    fprintf(stderr,"FITS error loading file: %s, no HDUs present\n",inFileName.c_str());
    fits_close_file(dataFile,&status);
    return EXIT_FAILURE;
  }

  fprintf(stdout,"FITS number of HDUs: %i\n",numHDUs);

  int hduType = 0;
  fits_get_hdu_type(dataFile,&hduType,&status);
  if (hduType != IMAGE_HDU) {
    fprintf(stderr,"FITS unsupported format, 1st HDU must be IMAGE\n");
    fits_close_file(dataFile,&status);
    return EXIT_FAILURE;
  }

  int naxis = 0;
  fits_get_img_dim(dataFile,&naxis,&status);
  if (naxis != 3) {
    fprintf(stderr,"FITS unsupported format, num dimensions must be 3\n");
    fits_close_file(dataFile,&status);
    return EXIT_FAILURE;
  }

  int bitpix = 0;
  long naxes[3];
  fits_get_img_param(dataFile,3,&bitpix,&naxis,naxes,&status);

  fprintf(stdout,"FITS img size is: (%i,%i,%i)\n",(int)naxes[0],(int)naxes[1],(int)naxes[2]);
  if (bitpix == BYTE_IMG) fprintf(stdout,"FITS data type is: BYTE\n");
  else if (bitpix == SHORT_IMG) fprintf(stdout,"FITS data type is: SHORT\n");
  else if (bitpix == LONG_IMG) fprintf(stdout,"FITS data type is: LONG\n");
  else if (bitpix == LONGLONG_IMG) fprintf(stdout,"FITS data type is: LONGLONG\n");
  else if (bitpix == FLOAT_IMG) fprintf(stdout,"FITS data type is: FLOAT\n");
  else if (bitpix == DOUBLE_IMG) fprintf(stdout,"FITS data type is: DOUBLE\n");

  if (bitpix != FLOAT_IMG) {
    fprintf(stderr,"FITS unsupported image format\n");
    fits_close_file(dataFile,&status);
    return EXIT_FAILURE;
  }

  std::vector<float> values(naxes[0]*size_t(naxes[1])*naxes[2]);

  long fpixel[3] = {1,1,1};
  int retVal = fits_read_pix(dataFile,TFLOAT,fpixel,naxes[0]*size_t(naxes[1])*naxes[2],0,
                             values.data(),0,&status);
  fprintf(stdout,"FITS retVal: %i, status: %i\n",retVal,status);

  std::string fileName = outFileBase + "_" + varName + "_"
    + std::to_string(naxes[0]) + "x"
    + std::to_string(naxes[1]) + "x"
    + std::to_string(naxes[2]) + "_float.raw";
  std::ofstream out(fileName,std::ios::binary);
  out.write((const char *)values.data(),values.size()*sizeof(values[0]));
  std::cout << "written to " << fileName << std::endl;

  fits_close_file(dataFile,&status);
  return EXIT_SUCCESS;
}
