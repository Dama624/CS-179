Hong Joon Park

Question 1: Common Errors
--------------------------------------------------------
--------------------------------------------------------

1.1
---------------------
What is wrong: Type error. The pointer “a” is declared as a pointer to an int, but the 3 is an integer. “int *a =“ is only compatible with addresses. 

We can fix the code by declaring a separate int variable:

void test1(){
    int b;
    int *a;
    b = 3;
    a = &b;
    *a = *a + 2;
    printf("%d",*a);
}


1.2
---------------------
What is wrong: The declaration for the pointers is wrong. 

We can fix the code through the following:

void test2(){
    int *a, *b;    // The syntax here is corrected.
    a = (int*) malloc(sizeof(int));
    b = (int*) malloc(sizeof(int));

    if (!(a && b)){
        printf("Out of memory");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}


1.3
---------------------
What is wrong: The malloc’s syntax is wrong.

We can fix the code through the following:

void test3(){
    // The following line’s syntax is corrected
    int i, *a = (int*) malloc(1000 * sizeof(int));

    if (!a){
        printf("Out of memory");
        exit(-1);
    }
    for (i = 0; i < 1000; i++) {
        *(i+a)=i;
    }
}


1.4
---------------------
What is wrong: Memory is allocated for the number of rows, but not for the number of columns.

We can fix the code through the following:

void test4(){
    int **a;
    int i;
    a = (int**) malloc(3*sizeof(int*));
    if (a == NULL) {
        printf(“Out of memory”);
        exit(-1);
    }
    for (i = 0; i < 100; i++){
        a[i] = (int*) malloc(100 * sizeof(int));
        if (a[i] == NULL) {
            printf(“Out of memory”);
            exit(-1);
        }
    }
    a[1][1] = 5;
}


1.5
---------------------
The condition for the if statement is incorrect. It is checking if there was no memory available for allocating the int.

We can fix the code through the following:

void test5(){
    int *a = (int*) malloc(sizeof(int));
    scanf("%d",a);
    if (*a == 0) // Check if the input from scanf == 0
        printf("Value is 0\n");
}


Question 2: Parallelization
--------------------------------------------------------
--------------------------------------------------------

2.1
---------------------
y_1[n] is easier and faster implementation on the GPU, because it is more easily parallelizable. y_2[n] depends on previous values (as shown through y_2[n-2] + y_2[n-1]). Values of y_1[n], on the other hand, are all independent of previous values, and so the values of x[n] can be calculated separately. 


2.2
---------------------
The EMA can be expanded to the form: y[n] = c*x[n] + (1-c)*x[n-1] + (1-c)^2*x[n-2] + … We see that x[n] has an exponentially decaying influence. If c is close to 1, (1-c) is close to 0, and for each successive x[n] term the constant factor becomes exponentially closer to 0. As such, to get a parallelizable approximation for c close to 1, we can ignore all terms after the first few.


Question 3: Small-kernel convolution
--------------------------------------------------------
--------------------------------------------------------

