  694  mkdir stable-diffusion-cpp
  695  cd stable-diffusion-cpp
  696  git clone --recursive https://github.com/leejet/stable-diffusion.cpp
  697  cd stable-diffusion.cpp
  698  ls
  699  mkdir build
  700  cd build
  701  cmake ..
  702  brew install cmake
  703  cmake ..
  704  cmake --build . --config Release
