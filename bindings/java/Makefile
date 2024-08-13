INCLUDE_DIRS = ../../src ../../blst/bindings

TARGETS=ckzg_jni.c ../../src/ckzg.c ../../lib/libblst.a

OPTIMIZATION_LEVEL=-O2

CC_FLAGS += \
	-pedantic \
	-Wall \
	-Wextra \
	-Waggregate-return \
	-Walloca \
	-Warray-bounds \
	-Wbad-function-cast \
	-Wc++-compat \
	-Wc++11-compat \
	-Wcast-align \
	-Wcast-qual \
	-Wconversion \
	-Wdisabled-optimization \
	-Wdouble-promotion \
	-Wenum-compare \
	-Wfloat-equal \
	-Wformat-nonliteral \
	-Wformat-security \
	-Wformat-y2k \
	-Wformat=2 \
	-Wframe-larger-than=1048576 \
	-Wimplicit \
	-Wimplicit-fallthrough \
	-Winit-self \
	-Winline \
	-Winvalid-pch \
	-Wlarger-than=512 \
	-Wlong-long \
	-Wmissing-declarations \
	-Wmissing-field-initializers \
	-Wmissing-format-attribute \
	-Wmissing-include-dirs \
	-Wmissing-noreturn \
	-Wmissing-prototypes \
	-Wnested-externs \
	-Wnull-dereference \
	-Wold-style-definition \
	-Woverlength-strings \
	-Wpacked \
	-Wpadded \
	-Wpointer-arith \
	-Wredundant-decls \
	-Wredundant-move \
	-Wshadow \
	-Wsign-compare \
	-Wsign-conversion \
	-Wstack-protector \
	-Wstrict-aliasing=2 \
	-Wstrict-overflow=5 \
	-Wstrict-prototypes \
	-Wswitch-default \
	-Wswitch-enum \
	-Wtype-limits \
	-Wundef \
	-Wuninitialized \
	-Wunreachable-code \
	-Wvariadic-macros \
	-Wwrite-strings

ifeq ($(OS),Windows_NT)
  CLANG_EXECUTABLE=gcc
  CLANG_FLAGS=-shared
  JNI_INCLUDE_FOLDER=win32
  OS_ARCH=amd64
  LIBRARY_RESOURCE=ckzg4844jni.dll
  GRADLE_COMMAND=gradlew
else
  CLANG_EXECUTABLE=clang
  UNAME_S := $(shell uname -s)
  UNAME_M := $(shell uname -m)
  ifeq ($(UNAME_S),Linux)
    CLANG_FLAGS=-fPIC -shared
    JNI_INCLUDE_FOLDER=linux
    ifeq ($(UNAME_M),x86_64)
        OS_ARCH=amd64
    else
        OS_ARCH=$(UNAME_M)
    endif
    LIBRARY_RESOURCE=libckzg4844jni.so
  endif
  ifeq ($(UNAME_S),Darwin)
    ifeq ($(JAVA_HOME),)
      JAVA_HOME := $(shell /usr/libexec/java_home)
    endif
    CLANG_FLAGS=-dynamiclib
    JNI_INCLUDE_FOLDER=darwin
    ifeq ($(UNAME_M),arm64)
        OS_ARCH=aarch64
    else
        OS_ARCH=$(UNAME_M)
    endif
    LIBRARY_RESOURCE=libckzg4844jni.dylib
  endif
  GRADLE_COMMAND=./gradlew
endif

LIBRARY_FOLDER=src/main/resources/ethereum/ckzg4844/lib/${OS_ARCH}

ifeq ($(JAVA_HOME),)
  $(error JAVA_HOME is not set and autodetection failed)
endif

all: build test

.PHONY: build
build:
	mkdir -p ${LIBRARY_FOLDER}
	${CLANG_EXECUTABLE} ${CC_FLAGS} ${CLANG_FLAGS} ${OPTIMIZATION_LEVEL} -Wall -Wextra -Werror -Wno-missing-braces -Wno-unused-parameter -Wno-format ${addprefix -I,${INCLUDE_DIRS}} -I"${JAVA_HOME}/include" -I"${JAVA_HOME}/include/${JNI_INCLUDE_FOLDER}" -o ${LIBRARY_FOLDER}/${LIBRARY_RESOURCE} ${TARGETS}

.PHONY: test
test:
	${GRADLE_COMMAND} clean check
