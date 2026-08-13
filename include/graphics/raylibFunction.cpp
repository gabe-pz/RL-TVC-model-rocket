#include "raylibFunction.h" 

Camera3D raylibInit(int FPS){
    //*****RAYLIB INITALIZATION*****
    //viewing screen
    int screenWidth  = 3500;
    int screenHeight = 2000;
    InitWindow(screenWidth, screenHeight, "tvc-model-rocket-sim");

    //raylib syncing things
    SetTargetFPS(FPS); 

    //raylib camera 
    Camera3D camera = { 0 };
    camera.position   = (Vector3){ 10.0f, 10.0f, 10.0f };  // Camera position
    camera.target     = (Vector3){ 0.0f, 0.0f, 0.0f };     // Camera looking-at point
    camera.up         = (Vector3){ 0.0f, 1.0f, 0.0f };     // Camera up vector
    camera.fovy       = 45.0f;                             // Field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;
    DisableCursor();    

    return camera;
}
void raylibDrawRocket(const double distanceToThrustVector, const double centerOfGravity, const std::array<double, 4>& stateQ, const std::array<double, 3> position, Camera3D& camera){
    //raylib rocket Dimensions
    float bodyRadius = 0.25f;
    float bodyHeight = 2.5f;
    float coneHeight = 0.7;
    float coneTopRadius = 0.05f; 
    float cgFrac = (distanceToThrustVector - centerOfGravity) / distanceToThrustVector;                             

    //coordinate transformation matrix between raylib vecs and my own vecs. 
    //Form for 4x4 matrix is: r1 u, r2 v, r3 w, r4 t, where u, v, and w is my x, y, and z hat vector mapped into raylibs coordinate system, and t is origin offset
    float cf[16] = {
        0,0,1,0,
        1,0,0,0,
        0,1,0,0,
        0,0,0,1
    };

    //rotation matrix creation
    float rf[16];
    quatToMat(stateQ, rf);

    //Apply Update
    UpdateCamera(&camera, CAMERA_FREE);

    //focus on vectorisZ
    if (IsKeyPressed(KEY_Z)){
        //my x, i.e position[0] is mapped into raylibs z, and so on
        camera.target = (Vector3){(float)position[1], (float)position[2], (float)position[0]};
    }
    

    BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(camera);

            rlPushMatrix();

                //transform into raylibs coords and move into rocket frame to then get rotation
                rlMultMatrixf(cf);                                   
                rlTranslatef(position[0], position[1], position[2]); 
                rlMultMatrixf(rf);                                   

                //rotate rocket to be facing correct way
                rlRotatef(90.0f, 1.0f, 0.0f, 0.0f);                

                //shift cyclinder down, such that c.g is point rotate about. i.e make it the origin
                rlTranslatef(0.0f, -cgFrac * bodyHeight, 0.0f);
                
                //main rocket body
                DrawCylinder((Vector3){0,0,0}, bodyRadius, bodyRadius, bodyHeight, 64, BLACK);

                //nose cone
                rlPushMatrix();
                    rlTranslatef(0.0f, bodyHeight, 0.0f);
                    DrawCylinder((Vector3){0,0,0}, coneTopRadius, bodyRadius, coneHeight, 64, RED);
                rlPopMatrix();

            rlPopMatrix();

            DrawGrid(100, 1.0f);
        
        EndMode3D();
    EndDrawing();
}