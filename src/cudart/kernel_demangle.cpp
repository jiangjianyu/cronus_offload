
#include <demangle.h>
#include <iostream>
#include <vector>

#include "kernel_demangle.h"
#include "debug.h"

char* kernel_name_parameter(const char* s) {
	auto name = cplus_demangle(s, DMGL_PARAMS | DMGL_AUTO);
	auto cur = name;
	std::vector<char> matching_stack;
	std::vector<std::string> parameters;

	auto last_name = cur;
	auto par_cnt = 0;

	while (*cur != '\0') {
		switch (*cur) {
			case '<': matching_stack.push_back('<'); break;
			case '>':
				if (matching_stack.back() == '<') {
					matching_stack.pop_back();
				} else {
					log_err("error!!");
				}
				break;
			case '(': 
				matching_stack.push_back('(');
				if (matching_stack.size() == 1) {
					std::cout << "par starting here: " << cur << "\n";
					last_name = cur + 1;
				}
				break;
			case ')':
				if (matching_stack.back() == '(') {
					matching_stack.pop_back();
				} else {
					log_err("error!!");
				}
				if (matching_stack.size() == 0) {
					auto s = std::string(last_name, cur - last_name);
					parameters.push_back(s);
					log_info("parameter index-%d: %s",  par_cnt++, s.c_str());
					// std::cout << "par ending here: " << cur << "\n";
				}
				break;
			case ',':
				if (matching_stack.size() == 1 && matching_stack.back() == '(') {
					// top-level new parameter
					auto s = std::string(last_name, cur - last_name);
					parameters.push_back(s);
					log_info("parameter index-%d: %s",  par_cnt++, s.c_str());
					last_name = cur + 1;
				}
				break;
			case ' ':
				if (cur - last_name == 0) {
					last_name = cur + 1;
				}
				break;
			default:
				break;
		}
		cur++;
	}
	char *ret = (char*)malloc(parameters.size() + 1);
	for (int i = 0;i < parameters.size();i++) {
		if (parameters[i].back() == '*')
			ret[i] = '*';
		else
			ret[i] = '$';
	}
	ret[parameters.size()] = '\0';
	return ret;
}