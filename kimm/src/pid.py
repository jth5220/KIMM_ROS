class PID():
    def __init__(self, kp, ki, kd, offset = 0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.offset = offset
        
        self.windup_guard = 5
        
        self.integral = 0
        self.error_prev = 0
        # self.time_prev = time.perf_counter()
        return
    
    def update(self, setpoint, measurement):
        # PID calculations
        error = setpoint - measurement
        
        # time_interval = time.perf_counter() - self.time_prev
        time_interval = 0.1
        
        P = error
        
        # self.integral += error
        self.integral += error*(time_interval)
        if error < 0:
            self.integral = 0
            
        if self.integral > self.windup_guard:
            self.integral = self.windup_guard
        # elif self.integral > self.windup_guard:
        #     self.integral = self.windup_guard
        
        # D = (error - self.error_prev)
        D = (error - self.error_prev)/time_interval
        
        p_term = self.kp*P
        i_term = self.ki*self.integral
        d_term = self.kd*D
        # calculate manipulated variable - MV 
        output = self.offset + p_term + i_term + d_term
        
        self.error_prev = error
        return output, (p_term, i_term, d_term)