#include "./bernstein_poly_approx.h"

string bernsteinPolyGeneration(char const *module_name, char const *function_name, char const *degree_bound, char const *box)
{
	PyObject *pName, *pModule, *pFunc;
	PyObject *pArgs, *pValue;
	int i;

	Py_Initialize();
	PyRun_SimpleString("import sys, os");
	PyRun_SimpleString("sys.path.append(\".\")");
	PyRun_SimpleString("print(sys.path)");

	pName = PyUnicode_DecodeFSDefault(module_name);
	/* Error checking of pName left out */

	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	if (pModule != NULL) {
		pFunc = PyObject_GetAttrString(pModule, function_name);
		/* pFunc is a new reference */

		if (pFunc && PyCallable_Check(pFunc)) {
			pArgs = PyTuple_New(2);

			PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(degree_bound));
			PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(box));

			pValue = PyObject_CallObject(pFunc, pArgs);
			Py_DECREF(pArgs);
			if (pValue != NULL) {
				//cout << "Result of call: " << PyUnicode_AsUTF8(pValue) << endl;
				return PyUnicode_AsUTF8(pValue);
				Py_DECREF(pValue);
			}
			else {
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr, "Call failed\n");
				return "1";
			}
		}
		else {
			if (PyErr_Occurred())
				PyErr_Print();
			cout << "Cannot find function: " << function_name << endl;
		}
		Py_XDECREF(pFunc);
		Py_DECREF(pModule);
	}
	else {
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", module_name);
		return "1";
	}
	if (Py_FinalizeEx() < 0) {
		return "120";
	}
	return "0";
}
