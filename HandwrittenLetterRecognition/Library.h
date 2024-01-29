// MyLibrary.h
#ifdef MYLIBRARY_EXPORTS
#define MYLIBRARY_API __declspec(dllexport)
#else
#define MYLIBRARY_API __declspec(dllimport)
#endif

#ifndef MY_LIBRARY_H
#define MY_LIBRARY_H

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) void trainAndEvaluateMLP();
	__declspec(dllexport) void trainAndEvaluateLinearModel();
	__declspec(dllexport) void trainAndEvaluateSVM();
	__declspec(dllexport) void testSimpleLinearSVM();
	__declspec(dllexport) void testSimple3DSVM();

#ifdef __cplusplus
}
#endif

#endif // MY_LIBRARY_H

