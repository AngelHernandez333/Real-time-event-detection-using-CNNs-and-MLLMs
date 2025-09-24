#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <string.h>
// Juego del caballo
const int max = 5;
const int movx[8] = {1, 2, 2, 1, -1, -2, -2, -1};
const int movy[8] = {-2, -1, 1, 2, 2, 1, -1, -2};

void CrearTablero(int mat[max][max])
{
    int i = 0, j = 0;
    while (i < max)
    {
        while (j < max)
        {
            mat[i][j] = 0;
            j++;
        }
        i++;
        j = 0;
    }
}
void Mostrar(int mat[max][max])
{
    int i = 0, j = 0;
    while (i < max)
    {
        while (j < max)
        {
            if (mat[i][j] < 10)
            {
                std::cout << " " << mat[i][j] << "  ";
            }
            else
            {
                std::cout << " " << mat[i][j] << " ";
            }
            j++;
        }
        i++;
        j = 0;
        std::cout << std::endl;
    }
}

int main()
{
    int mat[max][max];
    int fila = 2, columna = 2;

    CrearTablero(mat);
    mat[fila][columna] = 1;
    // Jugar(mat, 2, columna, fila);
    Mostrar(mat);
    return 0;
}