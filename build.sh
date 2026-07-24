#!/bin/bash -l

PROJDIR="$(dirname $(realpath $0))"
SRCDIR="$PROJDIR/tests"

cd $PROJDIR

echo
echo "------------------------------------------------------------------------------------------------------------------------"
echo "                                            ********** Build Test **********"
echo

CC=gcc

CFLAGS="-Wall -Werror -Wno-unknown-pragmas -Wno-unknown-attributes -Wno-unknown-warning-option -march=native -fPIC -std=c17"
CFLAGS="$CFLAGS -D_DEFAULT_SOURCE"
CFLAGS="$CFLAGS -I$SRCDIR"

LDLIBS="-lm"

if [ "$#" -gt 0 -a "$1" = "debug" ]
then
    echo "debug build"
    CFLAGS="$CFLAGS -O0 -DCOY_PROFILE"
    CFLAGS="$CFLAGS -g -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function"
elif [ "$#" -gt 0 -a "$1" != "clean" -o \( "$#" = 0 \) ]
then
    echo "release build"
    CFLAGS="$CFLAGS -O3 -DNDEBUG -DCOY_PROFILE"
fi

if [ "$#" -gt 0 -a "$1" = "clean" ] 
then
    echo "clean compiled test program and test output"
    echo
    rm -rf test profile 
fi

if [ "$#" -gt 0 -a "$1" = "debug" ]
then
    echo
    echo "build test"
    $CC $CFLAGS $SRCDIR/test.c -o test $LDLIBS
    echo "build profile"
    $CC $CFLAGS $SRCDIR/profile.c -o profile $LDLIBS

    if [ "$#" -gt 1 -a "$2" = 'test' ]
    then
        echo "run test" && echo && echo && ./test && echo && ./profile
    fi
fi

if [ "$#" -gt 0 -a "$1" = "test" ]
then
    echo
    echo "build test"
    $CC $CFLAGS $SRCDIR/test.c -o test $LDLIBS
    echo "build profile"
    $CC $CFLAGS $SRCDIR/profile.c -o profile $LDLIBS && echo && echo "run test" && echo && echo && ./test && echo && ./profile
fi

if [ "$#" -gt 0 -a "$1" = "prof" ]
then
    echo
    echo "build profile"
    $CC $CFLAGS $SRCDIR/profile.c -o profile $LDLIBS && echo && echo && ./profile
fi

echo
echo "                                              ********** Done Test **********"
echo "------------------------------------------------------------------------------------------------------------------------"
echo

