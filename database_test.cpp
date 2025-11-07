#include "database.hpp"
#include <stdio.h>
#include <iostream>
int main()
{
	std::cout << "Testing db: " << std::endl;
	database::Database mydb = database::Database();
	std::cout << "Done" << std::endl;
}
