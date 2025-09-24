#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void MostrarDiccionario(std::vector<std::vector<std::string>> dic)
{
    for (int i = 0; i < dic.size(); i++)
    {
        std::cout << dic[i][0] << ":" << dic[i][1];
        std::cout << std::endl;
    }
}
std::string findValue(std::string key, std::vector<std::vector<std::string>> dic)
{
    for (int i = 0; i < dic.size(); i++)
    {
        if (dic[i][0] == key)
        {
            return dic[i][1];
        }
    }
    return "No encontrado";
}

std::string findIndex(int index, std::vector<std::vector<std::string>> dic)
{
    if (index >= 0 && index < dic.size())
    {
        return dic[index][1];
    }
    return "No encontrado";
}
void appendValue(std::string key, std::string value, std::vector<std::vector<std::string>> &dic)
{
    dic.push_back({key, value});
}
int main()
{
    std::vector<std::vector<std::string>> dictionario = {
        {"Programacion", "proceso de componer y organizar un conjunto de instrucciones para la creacion de software"},
        {"Algoritmo", "conjunto de instrucciones definidas y acotadas para resolver un problema"},
        {"Estructuras de control", "conjunto de reglas que permiten controlar el flujo de ejecucion del programa"},
        {"Codigo", "Conjunto de instrucciones que un desarrollador ordena ejecutar a una computadora"},
        {"Lenguages de programacion", "Lenguaje informatico especiamente dise;adop para describir el conjunto de acciones consecutivas o instrucciones que un equpo informatico puede ejecutar"},
    };

    MostrarDiccionario(dictionario);
    std::string palabra = "Codigo";
    std::cout << "Estado de : " << palabra << " : " << findValue(palabra, dictionario) << std::endl;
    int indice = 2;
    std::cout << "Estado de : " << indice << " : " << findIndex(indice, dictionario) << std::endl;

    appendValue("Nuevo termino", "Definicion del nuevo termino", dictionario);
    MostrarDiccionario(dictionario);
    palabra = "Nuevo termino";
    std::cout << "Estado de : " << palabra << " : " << findValue(palabra, dictionario) << std::endl;

    return 0;
}