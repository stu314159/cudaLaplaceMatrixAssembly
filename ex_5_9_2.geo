//mesh for Kwon book example 5.9.2 which is an exercise of the Laplace
//equation over a simple rectangular domain.

lc = .05; 

Point(1)={0,0,0,lc};
Point(2)={5,0,0,lc};
Point(3)={5,10,0,lc};
Point(4)={0,10,0,lc};

Line(1)={1,2};
Line(2)={2,3};
Line(3)={3,4};
Line(4)={4,1};

Line Loop(5)={1,2,3,4};

Plane Surface(6)={5};

Physical Line(7)={1}; //bottom
Physical Line(8)={2}; //right
Physical Line(9)={3}; //top
Physical Line(10)={4}; //left

Physical Surface(11)={6}; //whole domain.