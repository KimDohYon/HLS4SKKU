# ==== 설정 ====
TOP := compute_attention_HLS
PLATFORM := /home/andrew/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm

KERNEL := $(TOP)
XO := $(KERNEL).xo
XCLBIN := $(KERNEL).xclbin
HOST := host
SRC := top.cpp
TB := host.cpp

# ==== 컴파일 플래그 ====
VPPFLAGS := --platform $(PLATFORM) -t hw -g --save-temps
CXXFLAGS := -std=c++17 -O0 -g -Wall

# ==== 전체 빌드 ====
all: $(XO) $(XCLBIN) $(HOST)

# ==== xo 생성 ====
$(XO): $(SRC)
	v++ -c $(VPPFLAGS) -k $(KERNEL) -o $@ $<

# ==== xclbin 생성 ====
$(XCLBIN): $(XO)
	v++ -l $(VPPFLAGS) -o $@ $<

# ==== host 코드 컴파일 ====
$(HOST): $(TB)
	g++ $(CXXFLAGS) \
		-I$(XILINX_XRT)/include \
		-I$(XILINX_VITIS)/include \
		-L$(XILINX_XRT)/lib \
		-lOpenCL -lpthread -lrt -lstdc++ \
		-o $@ $<

# ==== 청소 ====
clean:
	rm -rf $(XO) $(XCLBIN) $(HOST) *.log .Xil _x *.o
