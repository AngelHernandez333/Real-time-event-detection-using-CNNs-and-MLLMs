#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void IniciarArreglo(std::vector<int> &arr)
{
    for (int i = 0; i < 100; i++)
    {
        arr.push_back(rand() % 100);
    }
}
void Mostrar(std::vector<int> arr)
{
    for (int i = 0; i < arr.size(); i++)
    {
        std::cout << i << "-" << arr[i] << std::endl;
    }
}
int Linear_Search(std::vector<int> arr, int value)
{
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] == value)
        {
            return i;
        }
    }
    return -1;
}
int Binary_Search(std::vector<int> arr, int value)
{
    int LimiteSuperior = arr.size() - 1;
    int LimiteInferior = 0;
    int check_value = (LimiteSuperior - LimiteInferior) / 2;
    while ((check_value < LimiteSuperior) && (check_value > LimiteInferior))
    {
        std::cout << check_value << "-" << LimiteInferior << "-" << LimiteSuperior << std::endl;
        if (arr[check_value] == value)
        {
            return check_value;
        }
        else if (arr[check_value] > value)
        {
            LimiteSuperior = check_value;
        }
        else if (arr[check_value] < value)
        {
            LimiteInferior = check_value;
        }
        check_value = ((LimiteSuperior - LimiteInferior) / 2) + LimiteInferior;
    }
    return -1;
}
void Binary_Search_Curso(std::vector<int> mat, int num)
{
    int min = 0;
    int max = mat.size() - 1;
    int mitad;
    while (min != max)
    {
        mitad = ((min + max) / 2);
        std::cout << mitad << std::endl;
        if (mat[mitad] == num)
        {
            std::cout << "\nnumero encontrado\n";
            return;
        }
        if (num < mat[mitad])
        {
            max = mitad - 1;
        }
        if (num > mat[mitad])
        {
            min = mitad + 1;
        }
    }
    std::cout << "\nnumero no encontrado\n";
    return;
}
void BubbleSort(std::vector<int> &mat)
{
    int cont = 0;
    bool sort = true;
    while (sort)
    {
        sort = false;
        for (int i = 0; i < mat.size() - 1; i++)
        {
            if (mat[i] > mat[i + 1])
            {
                std::swap(mat[i], mat[i + 1]);
                cont++;
                sort = true;
            }
        }
    }
    std::cout << std::endl
              << cont << std::endl;
}
void BubbleCurso(std::vector<int> &mat)
{
    int i = 0;
    int j = 0, aux;
    while (i < mat.size())
    {
        while (j < mat.size() - i)
        {
            if (mat[j] > mat[j + 1])
            {
                aux = mat[j + 1];
                mat[j + 1] = mat[j];
                mat[j] = aux;
            }
            j++;
        }
        i++, j = 0;
    }
}
void InsertionSort(std::vector<int> &mat)
{
    int cont = 0;
    for (int i = 0; i < mat.size() - 1; i++)
    {
        if (mat[i] > mat[i + 1])
        {
            std::swap(mat[i + 1], mat[i]);
            cont++;
            for (int j = i; j > 0; j--)
            {
                if (mat[j - 1] > mat[j])
                {
                    std::swap(mat[j], mat[j - 1]);
                    cont++;
                }
                else
                {
                    break;
                }
            }
        }
    }
    std::cout << "El numero de operaciones es " << cont << std::endl;
}
void SelectionSort(std::vector<int> &mat)
{
    for (int i = 0; i < mat.size() - 1; i++)
    {
        int index = i;
        int value = mat[i];
        for (int j = i + 1; mat.size() > j; j++)
        {
            if (mat[j] < value)
            {
                index = j;
                value = mat[j];
            }
        }
        if (index != i)
        {
            std::swap(mat[i], mat[index]);
        }
    }
}
int QuickSort(std::vector<int> &mat, int inicio, int fin)
{
    int piv, izq, der;
    if (inicio < fin)
    {
        piv = mat[fin];
        izq = inicio;
        der = fin;
    }
    while (izq < der)
    {
        while (mat[izq] < piv)
        {
            izq++;
        }
        mat[der] = mat[izq];
        der--;
        while (mat[der] > piv)
        {
            der--;
        }
        if (der > izq)
        {
            mat[izq] = mat[der];
            izq++;
        }
    }
    mat[der] = piv;
    QuickSort(mat, inicio, der - 1);
    QuickSort(mat, der + 1, fin);
}

int main()
{
    std::vector<int> arr;
    IniciarArreglo(arr);
    Mostrar(arr);
    // BubbleSort(arr);
    QuickSort(arr, 0, 99);
    std::cout << "-----------------------------" << std::endl;
    Mostrar(arr);
    return 0;
}