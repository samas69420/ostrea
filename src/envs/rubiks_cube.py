import gymnasium as gym
import torch
import numpy as np

i_to_col = {0:'r',1:'b',2:'o',3:'g',4:'y',5:'w'}

def _rot_mat_cw(sq_matrix):
    """
    a11 a12 ...      ... a21 a11
    a21 ...      ->      ... a12
    ...                      ...    

    """
    
    result = sq_matrix.copy()
    N = sq_matrix.shape[0]

    for i in range(N):
        result[:,N-1-i] = sq_matrix[i,:]

    return result

def _rot_mat_ccw(sq_matrix):

    """
    a11 a12 ...      ...
    a21 ...      ->  a12 ...
    ...              a11 a21 ...
                         
    """    

    result = sq_matrix.copy()
    N = sq_matrix.shape[0]

    for i in range(N):
        result[:,i] = np.flip(sq_matrix[i,:])

    return result

def _rot_cube_cw(cube3d):

    """
       4                cw0
     0 1 2 3  --->  cw5 cw1 cw4 ccw3 
       5                cw2    
    """ 
    
    result = cube3d.copy()
    result[0] = _rot_mat_cw(cube3d[5])
    result[1] = _rot_mat_cw(cube3d[1])
    result[2] = _rot_mat_cw(cube3d[4])
    result[3] = _rot_mat_ccw(cube3d[3])
    result[4] = _rot_mat_cw(cube3d[0])
    result[5] = _rot_mat_cw(cube3d[2])

    return result

def _rot_cube_ccw(cube3d):

    """
       4                 ccw2
     0 1 2 3  --->  ccw4 ccw1 ccw5 cw3 
       5                 ccw0    
    """ 
    
    result = cube3d.copy()
    result[0] = _rot_mat_ccw(cube3d[4])
    result[1] = _rot_mat_ccw(cube3d[1])
    result[2] = _rot_mat_ccw(cube3d[5])
    result[3] = _rot_mat_cw(cube3d[3])
    result[4] = _rot_mat_ccw(cube3d[2])
    result[5] = _rot_mat_ccw(cube3d[0])

    return result

def _rot_rows_in_cube_ccw(result, inp, i):
    """
       _  _  _
      |_||_||_|
      |<||<||<| i-th row
      |_||_||_|
      
    """
    result[0,i,:] = inp[1,i,:]
    result[1,i,:] = inp[2,i,:]
    result[2,i,:] = inp[3,i,:]
    result[3,i,:] = inp[0,i,:]

def _rot_rows_in_cube_cw(result, inp, i):
    """
       _  _  _
      |_||_||_|
      |>||>||>| i-th row
      |_||_||_|
      
    """
    result[0,i,:] = inp[3,i,:]
    result[3,i,:] = inp[2,i,:]
    result[2,i,:] = inp[1,i,:]
    result[1,i,:] = inp[0,i,:]

def _print_cube(cube3d, col=True):

    N = cube3d[0].shape[0]
    
    print()
    
    for i in range(N):
        face = 4
        print(" "*(N*2+3), end = '')
        for j in range(N):
            element = cube3d[face,i,j]
            if col:
                element = i_to_col[element]
            print(element, end=' ')
        print()

    print()

    for i in range(N):
        for face in range(4):
            for j in range(N):
                element = cube3d[face,i,j]
                if col:
                    element = i_to_col[element]
                print(element, end=' ')
            if face != 3:
                print(" | ",end='')
        print()

    print()

    for i in range(N):
        face = 5
        print(" "*(N*2+3), end = '')
        for j in range(N):
            element = cube3d[face,i,j]
            if col:
                element = i_to_col[element]
            print(element, end=' ')
        print()

    print()


class RubiksCube(gym.Env):

    metadata = {"render_modes": ["human"]}


    def __init__(self, N = 3, win_reward = 20., step_penalty = 0.1, **kwargs):
        super(RubiksCube).__init__()

        if 'render_mode' in kwargs:
            self.render_mode = kwargs["render_mode"]
        else:
            self.render_mode = None

        self.N = N
        self.goal_state = np.ones(shape=(6,N,N))*np.arange(6).reshape(-1,1,1)

        self.action_space = gym.spaces.Discrete(N*4)
        self.observation_space = gym.spaces.Box(low=0,high=1,shape=(6*N*N,),dtype=np.float64)
        self.scramble_steps = 3
        self.step_penalty = step_penalty
        self.win_reward = win_reward 
        self.tot_ep = 0
        self.tot_steps = 0


    def _scramble(self):

        for _ in range(self.scramble_steps):
            random_rot = self.action_space.sample()
            #print(f"scramble: {random_rot}")
            self._execute_rotation(random_rot)


    def _execute_rotation(self, rotation):
        """
                 0  1  2
                 v  v  v
           11 > |_||_||_| < 3
           10 > |_||_||_| < 4
            9 > |_||_||_| < 5
                 ^  ^  ^ 
                 8  7  6
        """

        new_state = self.state.copy()

        # handle the 8 corners

        if rotation == 0:

            new_state = _rot_cube_cw(new_state)
            orig_rot = new_state.copy()
            new_state[4] = _rot_mat_cw(new_state[4])
            _rot_rows_in_cube_ccw(new_state, orig_rot, 0)
            new_state = _rot_cube_ccw(new_state)

        elif rotation == self.N-1:

            new_state = _rot_cube_cw(new_state)
            orig_rot = new_state.copy()
            new_state[5] = _rot_mat_ccw(new_state[5])
            _rot_rows_in_cube_ccw(new_state, orig_rot, self.N-1)
            new_state = _rot_cube_ccw(new_state)

        elif rotation == self.N:
            
            orig = self.state
            new_state[4] = _rot_mat_cw(new_state[4])
            _rot_rows_in_cube_ccw(new_state, orig, 0)

        elif rotation == 2*self.N-1:

            orig = self.state
            new_state[5] = _rot_mat_ccw(new_state[5])
            _rot_rows_in_cube_ccw(new_state, orig, self.N-1)

        elif rotation == 2*self.N:

            new_state = _rot_cube_ccw(new_state)
            orig_rot = new_state.copy()
            new_state[4] = _rot_mat_cw(new_state[4])
            _rot_rows_in_cube_ccw(new_state, orig_rot, 0)
            new_state = _rot_cube_cw(new_state)

        elif rotation == 3*self.N-1:

            new_state = _rot_cube_ccw(new_state)
            orig_rot = new_state.copy()
            new_state[5] = _rot_mat_ccw(new_state[5])
            _rot_rows_in_cube_ccw(new_state, orig_rot, self.N-1)
            new_state = _rot_cube_cw(new_state)

        elif rotation == 3*self.N:

            orig = self.state
            new_state[5] = _rot_mat_cw(new_state[5])
            _rot_rows_in_cube_cw(new_state, orig, self.N-1)

        elif rotation == 4*self.N-1:

            orig = self.state
            new_state[4] = _rot_mat_ccw(new_state[4])
            _rot_rows_in_cube_cw(new_state, orig, 0)

        # handle the central rows/cols

        # top side (like action 1 in the case of N=3)
        
        elif rotation > 0 and rotation < self.N-1:

            new_state = _rot_cube_cw(new_state)
            orig_rot = new_state.copy()
            _rot_rows_in_cube_ccw(new_state, orig_rot, rotation % self.N)
            new_state = _rot_cube_ccw(new_state)

        # right side (like action 4 in the case of N=3)

        elif rotation > self.N and rotation < 2*self.N-1:

            orig = self.state
            _rot_rows_in_cube_ccw(new_state, orig, rotation % self.N)

        # bottom side (like action 7 in the case of N=3)
        
        elif rotation > 2*self.N and rotation < 3*self.N-1:

            new_state = _rot_cube_ccw(new_state)
            orig_rot = new_state.copy()
            _rot_rows_in_cube_ccw(new_state, orig_rot, rotation % self.N)
            new_state = _rot_cube_cw(new_state)

        # left side (like action 10 in the case of N=3)

        elif rotation > 3*self.N and rotation < 4*self.N-1:

            orig = self.state
            _rot_rows_in_cube_cw(new_state, orig, self.N-(rotation % self.N)-1)

        self.state = new_state


    @property
    def observation(self):
        return self.state.flatten()/6.
        

    def step(self, action):

        self.tot_steps += 1
        terminated = False

        if self.render_mode == "human":
            self.render()
            print("action:",action, "steps: ", self.tot_steps)
            print()

        self._execute_rotation(action)

        reward = -self.step_penalty

        if (self.state == self.goal_state).all():
            terminated = True
            reward = self.win_reward
            self.reset()

        truncated = False

        return (self.observation, reward, terminated, truncated, {})


    def reset(self, seed = None, options = None):

        self.tot_ep += 1
        self.state = self.goal_state
        self._scramble()

        # increse difficulty every 100 episodes (curriculum learning)
        if self.tot_ep % 100 == 0:
            self.scramble_steps = np.min((self.scramble_steps+1,20))
            print("levelup to:", self.scramble_steps)

        return (self.observation, {})


    def render(self):

        _print_cube(self.state)


gym.register(
    id="RubiksCube",
    entry_point="envs.rubiks_cube:RubiksCube",
    max_episode_steps=500)

if __name__ == "__main__":

    N = 3

    cube = RubiksCube(N, render_mode = "human")
    cube.reset()
    breakpoint()

    #cube = RubiksCube(N)
    #for a in range(4*N):
    #    cube.reset()
    #    cube.step(a)
    #    cube.render()
    #    print(a)
    #    breakpoint()

