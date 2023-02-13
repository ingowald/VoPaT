#define STB_IMAGE_IMPLEMENTATION

#include "common/vopat.h"
#include "stb/stb_image.h"
#include <fstream>

using namespace vopat;

std::vector<float> readTif(const std::string &fileName,
                           vec2i &size)
{
  std::cout << "importing " << fileName << std::endl;
  int n;
  unsigned char *data = stbi_load(fileName.c_str(),&size.x,&size.y,&n,0);
  // PING;
  // PRINT(size);
  // PRINT(n);
  // PRINT(*(uint32_t**)data);
  // PRINT(*(float*)data);
  std::vector<float> ret;
  for (int i=0;i<size.x*size.y;i++)
    ret.push_back(data[i*n]/256.f);
  return ret;
}

int main(int ac, char **av)
{
  vec2i size;
  std::vector<std::vector<float>> images;
  for (int i=1;i<ac;i++)
    images.push_back(readTif(av[i],size));
  std::ofstream out("stacked.raw",std::ios::binary);
  for (auto &img : images)
    out.write((char*)img.data(),size.x*size.y*sizeof(float));
  std::cout << "written stacked.raw, dims are " << size.x << " " << size.y << " " << images.size() << std::endl;
}
