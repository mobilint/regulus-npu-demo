##################     Setting    ##################
OPENCV_INCLUDE = `pkg-config --cflags opencv4`
OPENCV_LDFLAGS = `pkg-config --libs opencv4`
MACCEL_INCLUDE = -I./maccel/include
MACCEL_LDFLAGS = -L./maccel/lib -lmaccel -Wl,-rpath=./maccel/lib
GSTREAMER_INCLUDE = `pkg-config --cflags gstreamer-1.0`
GSTREAMER_LDFLAGS = `pkg-config --libs gstreamer-1.0` -lgstapp-1.0

LDFLAGS = $(OPENCV_LDFLAGS) $(MACCEL_LDFLAGS) $(GSTREAMER_LDFLAGS)
INCLUDE = $(OPENCV_INCLUDE) $(MACCEL_INCLUDE) $(GSTREAMER_INCLUDE)

# 소스 파일 디렉토리
SRC_DIR = ./src
# 빌드 결과 디렉토리
BUILD_DIR = .
# 빌드 결과물 이름
TARGET_NAME = demo
# 오브젝트 파일 디렉토리
OBJ_DIR = ./obj
###################################################

TARGET = $(BUILD_DIR)/$(TARGET_NAME)

CXXFLAGS = -O3
LDFLAGS += -pthread -fopenmp
INCLUDE += -I./src

SRCS = $(notdir $(wildcard $(SRC_DIR)/*.cc))
OBJS = $(patsubst %.o,$(OBJ_DIR)/%.o,$(SRCS:.cc=.o))

.PHONY: all clean
all: $(TARGET)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cc
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

$(TARGET) : $(OBJS) $(SRC_DIR)/*
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
