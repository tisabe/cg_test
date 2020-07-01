

#ifndef GLOBAL_H
#define GLOBAL_H

#include <float.h>

#ifdef MAIN_PROGRAM
   #define EXTERN
#else
   #define EXTERN extern
#endif

/*
   Globale Variablen stehen in allen Funktionen zur Verfuegung.
   Achtung: Das gilt *nicht* fuer Kernel-Funktionen!
*/
EXTERN int npts;
EXTERN int gridsize, blocksize;

#undef EXTERN

#endif
