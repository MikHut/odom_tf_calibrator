#!/usr/bin/env python

from math import pi, exp

from odom_tf_calibrator.tf2d import TF2d#, Rotation

from scipy.optimize import minimize




def fix_rad_range( rad ):
    if rad > pi:
        rad -= 2.0*pi
    if rad < -1.0*pi:
        rad += 2.0*pi
    return rad




class Optimizer( object ):
    """ The Optimizer class tries to find the correct transformation between sensors. The first sensor acts as base line and all other sensors
    are aligned to it """
    def __init__( self ):
        #self.sensors = sensors
        self.log_filename = '/home/johann/rasberry_ws/src/local_testing/log/calib.txt'
        pass
    
    def xyt_loss( self, xyt_a, xyt_b):
        """ computes the error between two transformations, provided via x/y/theta parameters. we return the sum of square errors """
        dx = xyt_b[0] - xyt_a[0]
        dy = xyt_b[1] - xyt_a[1]
        dt = fix_rad_range( xyt_b[2] - xyt_a[2] )
        return dx*dx + dy*dy + dt*dt
    
    def compute_weights( self, data, x ):
        """ computes the weights based on the square error.
        - everything equal or lower than the median error has weight 1.0
        - everything with a square error higher or equal to 10x the median error has weight 0.0
        - everything in between has a weight computed by exponential decay, in the range (1.0, 0.0) """
        errors = self.compute_errors( data, x )
        for i in range(1, len(self.topics)):
            topic = self.topics[i]
            error_list = errors[topic]
            dp_list = data[topic]
            num_dp = len(dp_list)
            min_e, max_e, median_e, mean_e = self.error_stats( error_list )
            exp_lambda = -1.0 / median_e # lambda for the exponential decay
            for k in range(num_dp):
                if error_list[k] <= median_e:
                    dp_list[k].weight = 1.0
                elif error_list[k] >= 10.0 * median_e:
                    dp_list[k].weight = 0.0
                else:
                    de = error_list[k] - median_e
                    w = exp( exp_lambda * de )
                    dp_list[k].weight = w
            
    
    def fun( self, x, data ):
        """ this method is passed to the optimizer as the function that is to be minimized.
        
        x is a 1d array of parameters to be optimized.
        data is a dictionary with a list of xyt data for each sensor
        the return value is the error value """
        
        #params = chunks( x, 3 ) # list of tripples (3 parameters for each sensors' transformation)
        #param = list(params)
        #print( 'fun x={}'.format(x) )
        param = []
        for i in range(1, len(self.topics)):
            offset = (i-1)*3
            param.append( (x[offset], x[offset+1], x[offset+2]) )
        #print( 'fun param={}'.format(param) )
        
        loss_value = 0.0
        ref_data = data[self.topics[0]] # data from the reference sensor
        data_length = len( ref_data )
        for i in range(1, len(self.topics)):   # iterate over all sensors, except the first one (which acts as reference point for all other sensors)
            #offset = TF2d.from_xyt( *param[i-1] )
            offset = TF2d.from_xyt( *param[i-1] )
            inv_offset = offset.inverse()
            data_subset = data[ self.topics[i] ]
            
            for k in range( data_length ):
                base_xyt = ref_data[k]
                data_point = data_subset[k]
                sensor_xyt = (inv_offset * TF2d.from_xyt(data_point[0], data_point[1],data_point[2]) * offset).as_xyt()
                loss_value += self.xyt_loss( base_xyt, sensor_xyt ) * data_point.weight
        return loss_value
    
    def log_error( self, data, result ):
        """ writes the errors per data point to a log file """
        with open( self.log_filename, 'a' ) as log_file:
            log_file.write( '~~~~~~~~ ~~~~~~~~ ~~~~~~~~ ~~~~~~~~\n' )
            num_sensors = len(self.topics)
            data_size = len(data[self.topics[0]])
            log_file.write( 'num_sensors={}, data size={}\n'.format(num_sensors, data_size) )
            log_file.write( 'result.x={}\n'.format(result.x) )
            log_file.write( 'errors:\n' )
            errors = self.compute_errors( data, result.x )
            for i in range(1,num_sensors):
                topic = self.topics[i]
                min_e, max_e, median_e, mean_e = self.error_stats(errors[topic])
                log_file.write( '    {}: min={}, max={}, median={}, mean={}\n'.format(topic, min_e, max_e, median_e, mean_e) )
            log_file.write( 'error list:\n' )
            for k in range(data_size):
                #log_file.write( '    ' )
                for i in range(1, num_sensors):
                    topic = self.topics[i]
                    err = errors[topic][k]
                    log_file.write( '    {}'.format(err) )
                    log_file.write( ', dp={}'.format(str(data[topic][k])) )
                log_file.write( '\n' )
        
    
    def compute_errors( self, data, x ):
        """ computes the individual errors for each data point with respect to the transformation parameters in 'x' """
        param = []
        all_errors = dict()
        for i in range(1, len(self.topics)):
            offset = (i-1)*3
            param.append( (x[offset], x[offset+1], x[offset+2]) )
        ref_data = data[self.topics[0]] # data from the reference sensor
        data_length = len( ref_data )
        num_sensors = len(self.topics)
        print( 'compute_errors: data_length={}, num_sensors={}'.format( data_length, num_sensors ) )
        for i in range(1, num_sensors):   # iterate over all sensors, except the first one (which acts as reference point for all other sensors)
            errors = []
            #offset = TF2d.from_xyt( *param[i-1] )
            offset = TF2d.from_xyt( *param[i-1] )
            inv_offset = offset.inverse()
            data_subset = data[ self.topics[i] ]
            
            for k in range( data_length ):
                base_xyt = ref_data[k]
                tupple = data_subset[k]
                sensor_xyt = (inv_offset * TF2d.from_xyt(tupple[0], tupple[1],tupple[2]) * offset).as_xyt()
                errors.append( self.xyt_loss(base_xyt, sensor_xyt) )
            all_errors[self.topics[i]] = errors
        return all_errors
    
    def error_stats( self, error ):
        """ returns min, max, median, mean value of the provided list """
        sorted_error = sorted( error )
        length = len( error )
        median_index = int( length/2 )
        mean = sum(error) / length
        return sorted_error[0], sorted_error[-1], sorted_error[median_index], mean
    
    def optimize( self, data, initial_guess, topics ):
        """ data is a flat array of data points (x, y, t).
        initial_guess is a flat array of transformation parameters for each sensor (x,y,t).
        topics is a list of odom topic names to indicate the order in which data is to be read to match initial_guess """
        self.topics = topics
        result = minimize( fun=self.fun, x0=initial_guess, args=(data), method='SLSQP' )
        #print( 'result={}'.format( result ) )
        self.print_result( data, initial_guess, topics, result )
        #self.log_error( data, result )
        return result
    
    def print_result( self, data, initial_guess, topics, result ):
        print( '~~ optimizer ~~~~~~' )
        print( 'sensors:' )
        print( '  {} (reference sensor)'.format(topics[0]) )
        for i in range(1,len(topics)):
            print( '  {}'.format(topics[i]) )
            offset = 3*(i-1)
            print( '    initial guess={}'.format(initial_guess[offset:offset+3]) )
            print( '    final estimate={}'.format(result.x[offset:offset+3]) )
        print( '~~~~~~~~~~~~~~~~~~~' )
        #print( result )
        #print( '~~~~~~~~~~~~~~~~~~~' )


if __name__ == '__main__':
    print( 'standalone operation currently not supported' )


