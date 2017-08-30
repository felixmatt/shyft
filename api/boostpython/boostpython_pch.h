#if defined(_WINDOWS)
#pragma once
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)
#pragma warning (disable : 4503)
#endif


#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/handle.hpp>
//#include <boost/python/numeric.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/overloads.hpp>
#include <boost/operators.hpp>


#include "core/core_pch.h"

// experiment with python doc standard macro helpers
#define doc_intro(intro) intro  "\n"
#define doc_parameters() "\nParameters\n----------\n"
#define doc_parameter(name_str,type_str,descr_str) name_str  " : "  type_str  "\n\t"  descr_str  "\n"
#define doc_paramcont(doc_str) "\t"  doc_str  "\n"
#define doc_returns(name_str,type_str,descr_str) "\nReturns\n-------\n"  name_str  " : "  type_str  "\n\t" descr_str "\n"
#define doc_notes() "\nNotes\n-----\n"
#define doc_note(note_str) note_str  "\n"
#define doc_see_also(ref) "\nSee Also\n--------\n" ref  "\n"
#define doc_ind(doc_str) "\t"  doc_str
