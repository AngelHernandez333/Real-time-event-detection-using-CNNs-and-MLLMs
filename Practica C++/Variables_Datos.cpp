#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>

void IniciarValores(std::vector<int> &v)
{
    for (int i = 0; i < v.size(); i++)
    {
        v[i] = rand() % 1000;
    }
}
void MostrarValores(std::vector<int> v)
{
    for (int i = 0; i < v.size(); i++)
    {
        std::cout << v[i] << std::endl;
    }
}
int buscarMayor(std::vector<int> v)
{
    int mayor = v[0];
    for (int i = 1; i < v.size(); i++)
    {
        if (v[i] > mayor)
        {
            mayor = v[i];
        }
    }
    return mayor;
}
int buscarMenor(std::vector<int> v)
{
    int menor = v[0];
    for (int i = 1; i < v.size(); i++)
    {
        if (v[i] < menor)
        {
            menor = v[i];
        }
    }
    return menor;
}
int main()
{
    srand(time(NULL));
    std::vector<int> a(10);
    IniciarValores(a);
    MostrarValores(a);
    std::cout << "El mayor es: " << buscarMayor(a) << std::endl;
    std::cout << "El menor es: " << buscarMenor(a) << std::endl;
    return 0;
}