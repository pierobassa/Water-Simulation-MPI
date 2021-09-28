/*
Created by Piero Bassa thanks to Professor William Spataro @ University Of Calabria
Sustained for Parallel Algorithms and Distributed Systems. A/A 2020/2021
*/

#include <iostream>
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const float MinMass = 0.0001;
const float MaxMass = 0.5;
const float MaxCompress = 2;
const float MaxSpeed = 30;
const float MinFlow = 0.01;


const float MinDraw = 0.01; //For water colors based on mass
const float MaxDraw = 1.1;

const int map_dim = 40; //40 because 40 is divisibile by 2, 4 e 8. 3 Which are common number of cores of CPU's


const int AIR = 0;
const int GROUND = 1;
const int WATER = 2;

//Variables for MPI Parallelism
const int root = 0;
int numOfProcesses;

//Allegro settings
const int display_dim = 880;
ALLEGRO_DISPLAY* display;

//Coherence between the matrixes and the allegro display
const int blockNumberPerSide = map_dim;
const int spacing = display_dim / blockNumberPerSide;

//2d array (matrix) contiguous allocation
template <class T>
T** matrixAllocation(int N, int M) {
    if (N == 0)
        throw std::invalid_argument("number of rows is 0");
    if (M == 0)
        throw std::invalid_argument("number of columns is 0");

    T** ptr = nullptr;
    T* pool = nullptr;

    ptr = new T * [N];  // allocate pointers 
    pool = new T[N * M];  // allocate pool 

    // now point the row pointers to the appropriate positions in
    // the memory pool
    for (unsigned i = 0; i < N; ++i, pool += M)
        ptr[i] = pool;

    // Done.
    return ptr;
}

//fill vector with value
template <class T>
void fillVector(T* vector, int dimension, T value) {
    for (int i = 0; i < dimension; i++)
        vector[i] = value;
}

template <class T>
void deleteMatrix(T** mat) {
    if (mat == nullptr)
        return;
    free(mat[0]);
    free(mat);
}


void initContainerMap(int**& blocks, float**& mass, float**& new_mass) {
    for (int x = 0; x < map_dim; x++) {
        for (int y = 0; y < map_dim; y++) {
            blocks[x][y] = AIR;
        }
    }

    for (int i = 0; i < map_dim; i++) {
        blocks[i][0] = GROUND;
        blocks[i][map_dim - 1] = GROUND;
    }
    for (int j = 0; j < map_dim; j++) {
        blocks[0][j] = GROUND;
        blocks[map_dim - 1][j] = GROUND;
    }

    //INIT CENTRAL CONTAINER
    for (int i = map_dim / 2 - 3; i <= map_dim / 2 + 3; ++i) {
        blocks[map_dim / 2 - 2][i] = GROUND;
    }
    for (int j = map_dim / 2 - 2; j >= map_dim / 2 - 5; j--) {
        blocks[j][map_dim / 2 - 3] = GROUND;
    }
    for (int j = map_dim / 2 - 2; j >= map_dim / 2 - 5; j--) {
        blocks[j][map_dim / 2 + 3] = GROUND;
    }

    //More ground for water flow demonstration
    blocks[3][map_dim / 2 - 1] = GROUND;
    blocks[3][map_dim / 2 - 2] = GROUND;
    blocks[3][map_dim / 2 + 1] = GROUND;
    blocks[3][map_dim / 2 + 2] = GROUND;
    
    blocks[map_dim / 2 + 2][16] = GROUND;
    blocks[map_dim / 2 + 3][15] = GROUND;
    blocks[map_dim / 2 + 3][17] = GROUND;
    blocks[map_dim / 2 + 3][16] = GROUND;
    blocks[map_dim / 2 + 4][16] = GROUND;
    blocks[map_dim / 2 + 4][15] = GROUND;
    blocks[map_dim / 2 + 4][17] = GROUND;
    blocks[map_dim / 2 + 4][18] = GROUND;
    blocks[map_dim / 2 + 4][14] = GROUND;

    blocks[map_dim / 2 + 2][24] = GROUND;
    blocks[map_dim / 2 + 3][23] = GROUND;
    blocks[map_dim / 2 + 3][25] = GROUND;
    blocks[map_dim / 2 + 3][24] = GROUND;
    blocks[map_dim / 2 + 4][24] = GROUND;
    blocks[map_dim / 2 + 4][23] = GROUND;
    blocks[map_dim / 2 + 4][25] = GROUND;
    blocks[map_dim / 2 + 4][26] = GROUND;
    blocks[map_dim / 2 + 4][22] = GROUND;


    //GROUND BLOCKS FOR WATER FLOW
    for (int j = 15; j < 26; j++) {
        blocks[32][j] = GROUND;
    }
  

    blocks[1][map_dim / 2] = WATER;
    blocks[1][map_dim / 2 - 1] = WATER;
    blocks[1][map_dim / 2 + 1] = WATER;


    for (int x = 0; x < map_dim; x++) {
        for (int y = 0; y < map_dim; y++) {
            mass[x][y] = blocks[x][y] == WATER ? MaxMass : 0.0; //MaxMass if the cell has Water, 0.0 otherwise
            new_mass[x][y] = blocks[x][y] == WATER ? MaxMass : 0.0;
        }
    }
}

bool untilESCisPressed(int rank) {
    int buf = 0;
    if (rank == root) {
        ALLEGRO_KEYBOARD_STATE key_state;
        al_get_keyboard_state(&key_state);
        if (!al_key_down(&key_state, ALLEGRO_KEY_ESCAPE)) //root will check if the key esc is NOT pressed
            buf = INT_MAX; //if so, this variable's value will become INT_MAX
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&buf, 1, MPI_INT, root, MPI_COMM_WORLD); //the variable will be broadcasted to all processors
    
    return buf == INT_MAX; //if it's INT_MAX, the program will be stopped
}

float map(float inValue, float minInRange, float maxInRange, float minOutRange, float maxOutRange) {
    float x = (inValue - minInRange) / (maxInRange - minInRange);
    float result = minOutRange + (maxOutRange - minOutRange) * x;
    return result;
}

void waterColor(float m, int& r, int& g, int& b) {
    if (m < MinDraw)
        m = MinDraw;
    else if (m > MaxDraw)
        m = MaxDraw;


    r = 50;
    g = 50;

    if (m < 1) {
        b = int(map(m, 0.01, 1, 255, 200));
        r = int(map(m, 0.01, 1, 240, 50));
 ;
        if (r < 50)
            r = 50;
        else if (r > 240)
            r = 240;
        g = r;
    }
    else {
        b = int(map(m, 1, 1.1, 190, 140));
    }


    if (b < 140)
        b = 140;
    else if (b > 255)
        b = 255;
}

void print(int** blocks, float **mass) {
    int r, g, b;

    for (int i = 0; i < map_dim; i++) {
        for (int j = 0; j < map_dim; j++) {
            if (blocks[j][i] == WATER) {
                waterColor(mass[j][i], r, g, b);
                al_draw_filled_rectangle(i * display_dim / map_dim, j * display_dim / map_dim, i * display_dim / map_dim + display_dim / map_dim, j * display_dim / map_dim + display_dim / map_dim, al_map_rgb(r, g, b));
            }
            else if (blocks[j][i] == AIR) {

                al_draw_filled_rectangle(i * display_dim / map_dim, j * display_dim / map_dim, i * display_dim / map_dim + display_dim / map_dim, j * display_dim / map_dim + display_dim / map_dim, al_map_rgb(0, 0, 0));
            }
            else if (blocks[j][i] == GROUND) {
                al_draw_filled_rectangle(i * display_dim / map_dim, j * display_dim / map_dim, i * display_dim / map_dim + display_dim / map_dim, j * display_dim / map_dim + display_dim / map_dim, al_map_rgb(245, 245, 60));
            }
        }
    }

    al_flip_display();
    al_rest(0.05);
}

float get_stable_state_b(float total_mass) {
    if (total_mass <= 1) {
        return 1;
    }
    else if (total_mass < 2 * MaxMass + MaxCompress) {
        return (MaxMass * MaxMass + total_mass * MaxCompress) / (MaxMass + MaxCompress);
    }
    else {
        return (total_mass + MaxCompress) / 2;
    }
}


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

    int** blocks = nullptr;
    float** mass = nullptr;
    float** new_mass = nullptr;

    if (map_dim % numOfProcesses != 0 && rank == root) { //First condition checks if the matrix dimension is divisible by the number of processes chosen
        cout << "Error: please use a divider of " << map_dim << " as number of processes." << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == root) {
        blocks = matrixAllocation<int>(map_dim, map_dim);
        mass = matrixAllocation<float>(map_dim, map_dim);
        new_mass = matrixAllocation<float>(map_dim, map_dim);

        initContainerMap(blocks, mass, new_mass);

        al_init();
        al_init_primitives_addon();
        display = al_create_display(display_dim, display_dim);
        al_install_keyboard();
        print(blocks, mass);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int** subBlocks = matrixAllocation<int>(map_dim / numOfProcesses, map_dim);
    float** subMass = matrixAllocation<float>(map_dim / numOfProcesses, map_dim);
    float** subNewMass = matrixAllocation<float>(map_dim / numOfProcesses, map_dim);

    if (rank == root)
        MPI_Scatter(&blocks[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, &subBlocks[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, NULL, &subBlocks[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == root)
        MPI_Scatter(&mass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, &subMass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, NULL, &subMass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);

    if (rank == root)
        MPI_Scatter(&new_mass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, &subNewMass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);
    else
        MPI_Scatter(NULL, 0, NULL, &subNewMass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);


    float* upperVectorMass = (float*)malloc(sizeof(float) * map_dim);
    float* lowerVectorMass = (float*)malloc(sizeof(float) * map_dim);

    int* upperVectorBlocks = (int*)malloc(sizeof(int) * map_dim);
    int* lowerVectorBlocks = (int*)malloc(sizeof(int) * map_dim);



    while (untilESCisPressed(rank)) {

        //Timings for parallel average evolution time
        //auto start_time = MPI_Wtime();
        

        MPI_Request r;
        MPI_Status s;

        //SEND STAGE OF THE MASS BORDERS FOR EACH PROCESS TO SUBMASS
        if (rank != root) //Send to the previous process your first row
            MPI_Isend(&subMass[0][0], map_dim, MPI_INT, rank - 1, 11, MPI_COMM_WORLD, &r);

        if (rank != numOfProcesses - 1) //All processes besides the last process proceed to receive the row
            MPI_Recv(&lowerVectorMass[0], map_dim, MPI_INT, rank + 1, 11, MPI_COMM_WORLD, &s);
        else
            fillVector<float>(lowerVectorMass, map_dim, 0); //Last process fills with zeros

        if (rank != numOfProcesses - 1) //Same operation but this time we send our last row to the next process
            MPI_Isend(&subMass[map_dim / numOfProcesses - 1][0], map_dim, MPI_INT, rank + 1, 44, MPI_COMM_WORLD, &r);

        if (rank != root) //Receive the row besided the root process
            MPI_Recv(&upperVectorMass[0], map_dim, MPI_INT, rank - 1, 44, MPI_COMM_WORLD, &s);
        else
            fillVector<float>(upperVectorMass, map_dim, 0); //root process fills with zeros


        //SEND STAGE OF THE BLOCKS BORDERS FOR EACH PROCESS TO SUBBLOCKS
        if (rank != root) 
            MPI_Isend(&subBlocks[0][0], map_dim, MPI_INT, rank - 1, 11, MPI_COMM_WORLD, &r);

        if (rank != numOfProcesses - 1) 
            MPI_Recv(&lowerVectorBlocks[0], map_dim, MPI_INT, rank + 1, 11, MPI_COMM_WORLD, &s);
        else
            fillVector<int>(lowerVectorBlocks, map_dim, 0); 


        if (rank != numOfProcesses - 1) 
            MPI_Isend(&subBlocks[map_dim / numOfProcesses - 1][0], map_dim, MPI_INT, rank + 1, 44, MPI_COMM_WORLD, &r);

        if (rank != root)
            MPI_Recv(&upperVectorBlocks[0], map_dim, MPI_INT, rank - 1, 44, MPI_COMM_WORLD, &s);
        else
            fillVector<int>(upperVectorBlocks, map_dim, 0); 


        float Flow = 0;
        float remaining_mass;

        if (rank == 0) {
            subBlocks[1][map_dim / 2] = WATER;
            subMass[1][map_dim / 2] = MaxMass;
            subNewMass[1][map_dim / 2] = MaxMass;
        }

        

        //REGOLE DI WATER FLOW
        for (int x = 0; x < map_dim / numOfProcesses; x++) {
            for (int y = 0; y < map_dim; y++) {
                if (subBlocks[x][y] == GROUND)
                    continue;

                Flow = 0;
                remaining_mass = subMass[x][y];

                if (remaining_mass <= 0)
                    continue;

                //The block below this one
                if (x < map_dim / numOfProcesses - 1) {
                    if ((subBlocks[x + 1][y] != GROUND)) { 
                        Flow = get_stable_state_b(remaining_mass + subMass[x + 1][y]) - subMass[x + 1][y];

                        if (Flow > MinFlow) {
                            Flow *= 0.5; //leads to smoother flow
                        }

                        if (Flow > min(MaxSpeed, remaining_mass))
                            Flow = min(MaxSpeed, remaining_mass);
                        else if (Flow < 0)
                            Flow = 0;


                        subNewMass[x][y] -= Flow;
                        subNewMass[x + 1][y] += Flow;
                        remaining_mass -= Flow;
                    }
                }

                if (x == map_dim / numOfProcesses - 1 && rank != numOfProcesses - 1) {
                    if ((lowerVectorBlocks[y] != GROUND)) { 
                        Flow = get_stable_state_b(remaining_mass + lowerVectorMass[y]) - lowerVectorMass[y];
                        if (Flow > MinFlow) {
                            Flow *= 0.5; //leads to smoother flow
                        }
                        
                        if (Flow > min(MaxSpeed, remaining_mass))
                            Flow = min(MaxSpeed, remaining_mass);
                        else if (Flow < 0)
                            Flow = 0;

                        subNewMass[x][y] -= Flow;


                        remaining_mass -= Flow;
                    }
                }

                if (remaining_mass <= 0) continue;

                //Left
                if (subBlocks[x][y - 1] != GROUND) { 
                    //Equalize the amount of water in this block and it's neighbour
                    Flow = (subMass[x][y] - subMass[x][y - 1]) / 4;
                    if (Flow > MinFlow) {
                        Flow *= 0.5;
                    }
                    //contraining FLow between 0 and remaining_mass
                    if (Flow > remaining_mass)
                        Flow = remaining_mass;
                    else if (Flow < 0)
                        Flow = 0;


                    subNewMass[x][y] -= Flow;
                    subNewMass[x][y - 1] += Flow;
                    remaining_mass -= Flow;
                }

                if (remaining_mass <= 0) continue;


                //Right
                if (subBlocks[x][y + 1] != GROUND) { 
                    //Equalize the amount of water in this block and it's neighbour
                    Flow = (subMass[x][y] - subMass[x][y + 1]) / 4;
                    if (Flow > MinFlow) {
                        Flow *= 0.5;
                    }
                    
                    if (Flow > remaining_mass)
                        Flow = remaining_mass;
                    else if (Flow < 0)
                        Flow = 0;

                    subNewMass[x][y] -= Flow;
                    subNewMass[x][y + 1] += Flow;
                    remaining_mass -= Flow;
                }


                //Up. Only compressed water flows upwards.
                if (x > 1) {
                    if (subBlocks[x - 1][y] != GROUND) { 
                        Flow = remaining_mass - get_stable_state_b(remaining_mass + subMass[x - 1][y]);

                        if (Flow > min(MaxSpeed, remaining_mass))
                            Flow = min(MaxSpeed, remaining_mass);
                        else if (Flow < 0)
                            Flow = 0;

                        subNewMass[x][y] -= Flow;
                        subNewMass[x - 1][y] += Flow;
                        remaining_mass -= Flow;


                    }

                }

                if (x == 0 && rank != 0) {
                    if (upperVectorBlocks[y] != GROUND) { 
                        Flow = remaining_mass - get_stable_state_b(remaining_mass + upperVectorMass[y]);

                        if (Flow > min(MaxSpeed, remaining_mass))
                            Flow = min(MaxSpeed, remaining_mass);
                        else if (Flow < 0)
                            Flow = 0;

                        subNewMass[x][y] -= Flow;

                        remaining_mass -= Flow;
                    }
                }
            }
        }

        //Copy the new mass values to the mass array
        for (int x = 0; x < map_dim / numOfProcesses; x++) {
            for (int y = 0; y < map_dim; y++) {
                subMass[x][y] = subNewMass[x][y];
            }
        }

        for (int x = 0; x < map_dim / numOfProcesses; x++) {
            for (int y = 0; y < map_dim; y++) {
                //Skip ground blocks
                if (subBlocks[x][y] == GROUND) continue;
                //Flag/unflag water blocks
                if (subMass[x][y] > MinMass) {
                    subBlocks[x][y] = WATER;
                    
                    //EVOLUTION OF STATE TIMINGS
                    //auto end_time = MPI_Wtime();
                    //cout << end_time - start_time << endl;
                }
                else {
                    subBlocks[x][y] = AIR;
                }
            }
        }

        for (int y = 0; y < map_dim; y++) {
            if (upperVectorMass[y] > MinMass) {
                subBlocks[0][y] = WATER;
                subMass[0][y] = upperVectorMass[y];
                subNewMass[0][y] = upperVectorMass[y];
            }
        }

        
        MPI_Barrier(MPI_COMM_WORLD);

        

        if (rank == root)
            MPI_Gather(&subBlocks[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, &blocks[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);
        else
            MPI_Gather(&subBlocks[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, NULL, 0, MPI_INT, root, MPI_COMM_WORLD);

        if (rank == root)
            MPI_Gather(&subMass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, &mass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, root, MPI_COMM_WORLD);
        else
            MPI_Gather(&subMass[0][0], map_dim * map_dim / numOfProcesses, MPI_INT, NULL, 0, MPI_INT, root, MPI_COMM_WORLD);

        if (rank == root) {
            print(blocks, mass);
        }
    }


    deleteMatrix(blocks);
    deleteMatrix(mass);
    deleteMatrix(new_mass);
    deleteMatrix(subBlocks);
    deleteMatrix(subMass);
    deleteMatrix(subNewMass);
    free(upperVectorMass);
    free(lowerVectorMass);


    MPI_Finalize();

    return 0;
}