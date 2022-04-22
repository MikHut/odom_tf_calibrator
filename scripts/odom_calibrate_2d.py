#!/usr/bin/env python

from __future__ import print_function

import sys
import pickle
from math import pi, sqrt
from functools import partial
from threading import Thread

from odom_tf_calibrator.tf2d import TF2d
from odom_tf_calibrator.optimizer import Optimizer
from odom_tf_calibrator.republisher import Republisher
from odom_tf_calibrator.datapoint import DataPoint

from rasberry_core.update_robot_config_yaml import RobotConfig

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf, tf2_ros
from tf.transformations import euler_from_quaternion


""" copied from https://stackoverflow.com/a/312464 """
def chunks( lst, n ):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), n):
        yield lst[i:i + n]


def odom_to_xyt( msg ):
    """ takes an Odometry message and return x,y, and theta (yaw angle)"""
    pose = msg.pose.pose
    x = pose.position.x
    y = pose.position.y
    o = pose.orientation
    (roll, pitch, yaw) = euler_from_quaternion( [o.x, o.y, o.z, o.w] )
    theta = yaw
    return x, y, theta


def fix_rad_range( rad ):
    """ trims the angle back to [pi, -pi]. only works with moderate values. """
    if rad > pi:
        rad -= 2.0*pi
    if rad < -1.0*pi:
        rad += 2.0*pi
    return rad



class CalibratorNode( object ):
    """ The CalibrationNode class is running the ros node and collects all the necessary data. the data is stored in self.keyframes and passed on to self.optimizer. """
    def __init__( self ):
        rospy.init_node( 'lidar_calibrator', anonymous=False )
        print( 'sys.argv={}'.format(sys.argv)  )
        #self.odom_list = [ 'odom', 'sensor_odom' ]
        self.base_frame = rospy.get_param('base_frame', 'base_link')
        self.update_robot_config = rospy.get_param('~update_robot_config', False)
        robot_config_file = rospy.get_param('~robot_config_file', '')
        self.robot_config = RobotConfig(robot_config_file)
        self.odom_list = [ 'odometry/base_raw', 'odom_front_scan', 'odom_back_scan' ] # overwritten by self.apply_params!
        self.original_tf = self.get_original_tf() # initial guess: x1, y1, theta1, x2, y2, theta2 (two sensors)
        self.initial_guess =(0,0,0,0,0,0)#self.get_original_tf()
        self.odom_latest = dict()
        self.keyframes = dict()
        self.init_done = False
        self.num_opt_iterations = rospy.get_param('~num_opt_iterations', 4) # number of optimization runs with weighted datapoints
        self.num_opt_step_size = rospy.get_param('~num_opt_step_size', 20) # we start the optimization first when we reach this many keyframes and re-run optimization for every multiple of this
        self.max_opt_keyframes = rospy.get_param('~max_opt_keyframes', 120) # we stop collecting keyframes / optimization whe we have more than this many keyframes
        self.save_data = None
        self.save_at = 200
        self.ref_odom = None # set by self.apply_params
        self.subscribers = []
        self.apply_params()
        self.read_args()

        self.republisher = Republisher( ['/scanner_front/scan', '/scanner_back/scan'] ) #order must match self.odom_list, ignoring the reference odometry

        self.subscribe()
        print( 'CalibratorNode running' )


    def get_original_tf( self ):
        print('Getting initial guess from original scans')
        front_laser_frame_id = rospy.wait_for_message('/scanner_front/scan', LaserScan).header.frame_id
        back_laser_frame_id = rospy.wait_for_message('/scanner_back/scan', LaserScan).header.frame_id
        print(front_laser_frame_id, back_laser_frame_id)

        tf_listener = tf.TransformListener(cache_time=rospy.Duration(10))

        got_front_tf = False
        got_back_tf = False

        got_config = False
        while not rospy.is_shutdown() and not got_config:

            if not got_front_tf and front_laser_frame_id is not None:
                print("got front frame id of ", front_laser_frame_id)
                now=rospy.Time(0)
                try:
                    tf_listener.waitForTransform(self.base_frame, front_laser_frame_id, rospy.Time.now(), rospy.Duration(10.0))
                    (trans,rot) = tf_listener.lookupTransform(self.base_frame, front_laser_frame_id, rospy.Time.now())
                    front_x, front_y, front_z = trans
                    print(euler_from_quaternion(rot))
                    front_roll, front_pitch, front_yaw = euler_from_quaternion(rot)
                    got_front_tf = True
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf2_ros.TransformException):
                    continue

            if not got_back_tf and back_laser_frame_id is not None:
                print("got back frame id of ", back_laser_frame_id)
                now=rospy.Time(0)
                try:
                    tf_listener.waitForTransform(self.base_frame, back_laser_frame_id, rospy.Time.now(), rospy.Duration(10.0))
                    (trans,rot) = tf_listener.lookupTransform(self.base_frame, back_laser_frame_id, rospy.Time.now())
                    back_x, back_y, back_z = trans
                    back_roll, back_pitch, back_yaw = euler_from_quaternion(rot)
                    got_back_tf = True
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf2_ros.TransformException):
                    continue

            if got_front_tf and got_back_tf:
                got_config = True
        print('Got initial guess from original scans (front - back): ', front_x, front_y, front_yaw, back_x, back_y, back_yaw )

        return (front_x, front_y, front_yaw, back_x, back_y, back_yaw)


    def subscribe( self ):
        """ subscribes to the relevant ros topics """
        self.subscribers.append( rospy.Subscriber( self.ref_odom, Odometry, partial(self.odom_callback, self.ref_odom) ) )
        for topic in self.odom_list:
            print( 'subscribing to topic \'{}\''.format(topic) )
            self.odom_latest[topic] = None
            self.keyframes[topic] = []
            self.subscribers.append( rospy.Subscriber( topic, Odometry, partial(self.odom_callback, topic) ) )


    def apply_params( self ):
        """ reads parameters and applies them immediately. unlike self.read_args, this method focuses on ros parameters and not command line parameters. """
        #if not rospy.has_param( '~odom_topics' ):
        #    rospy.logerr( 'mandatory parameter \'odom_topics\' not set. (example value: \'sensor_a_odom,sensor_b_odom,sensor_c_odom\')' )
        #    rospy.signal_shutdown( 'error - shutting down node' )
        #    exit( 0 )
        #self.ref_odom = rospy.get_param( '~reference_odom', 'odometry' )
        topic_list = rospy.get_param( '~odom_topics', 'odometry/base_raw,odom_front_scan,odom_back_scan' )
        odom_topics = topic_list.split( ',' )
        self.ref_odom = odom_topics[0]
        self.odom_list = odom_topics
    
    def read_args( self ):
        """ reads command line arguments and applies them """
        for k in range(1, len(sys.argv)):
            arg = sys.argv[k]
            if arg.startswith('save='):
                self.save_data = arg[5:]
            if arg.startswith('save_at='):
                self.save_at = int(arg[8:])
                
    def odom_callback( self, topic, msg ):
        """ callback method for odometry messages. """
        xyt = odom_to_xyt( msg )
        self.odom_latest[topic] = xyt
        
        if not self.init_done:
            print( 'odom_callback init not finished yet' )
            for key in self.odom_latest:
                if self.odom_latest[key] == None:
                    return
            self.init_done = True
            print( 'odom_callback init finished' )
        
        self.check_for_keyframe( topic, xyt )
        
    def check_for_keyframe( self, topic, xyt ):
        """ checks if we have enough movement to create a new keyframe, and if so calls self.create_keyframe """
        if not self.init_done:
            return
        
        if not self.keyframes[topic]:
            print( 'check_for_keyframe: creating first keyframe' )
            self.create_keyframe()
            return
        
        odom = self.keyframes[topic][-1]
        dx = xyt[0] - odom[0]
        dy = xyt[1] - odom[1]
        dt = fix_rad_range( xyt[2] - odom[2] )
        dist = sqrt( dx*dx + dy*dy )
        if dist > 0.3 or dt > pi*35.0/180.0:    # 0.3m traveled (line of sight) or 35 degree bearing change
            #num_keyframes = len( self.keyframes[ self.odom_list[0] ] )
            #print( 'keyframe #{} dist={}, dt={}'.format(num_keyframes, dist, dt) )
            self.create_keyframe()
    
    def create_keyframe( self ):
        """ creates a new keyframe for all registered odometry sources. initiates optimization run if number of keyframes
        is multiple of self.num_opt_step_size """
        for key in self.odom_latest:
            self.keyframes[key].append( self.odom_latest[key] )
            
        num_keyframes = len( self.keyframes[ self.odom_list[0] ] )
        if num_keyframes % self.num_opt_step_size == 0 and num_keyframes <= self.max_opt_keyframes:
            new_thread = Thread( target=self.optimize )
            new_thread.start()
        if num_keyframes >= self.max_opt_keyframes:
            for sub in self.subscribers:
                sub.unregister()
            self.subscribers = []
    
    def optimize( self ):
        """ starts the optimization process. we first run the optimization process, then computes individual weights,
        and runs the optimization process again. """
        data = self.create_data()

        # initial optimization run
        initial_guess = self.initial_guess
        opt = Optimizer()
        result = opt.optimize( data, initial_guess, self.odom_list )

        # optimization loop with weighted datapoints
        for i in range(self.num_opt_iterations):
            initial_guess = result.x
            opt.compute_weights( data, result.x )
            result = opt.optimize( data, initial_guess, self.odom_list )
        self.initial_guess = result.x
        self.republisher.update_tf( [ self.original_tf[0], self.original_tf[1], result.x[2], self.original_tf[3], self.original_tf[4], result.x[5] ] )

        print("Calibrated angles: ", result.x[2], result.x[5] - pi )

        if self.update_robot_config:
            print("UPDATING CONFIG---------------------")
            print("data types: ", type(float(result.x[2])), type(float(result.x[5] - pi)))
            self.robot_config.update_entry('lidar_front', ['yaw'], [float(result.x[2])])
            self.robot_config.update_entry('lidar_back', ['yaw'], [float(result.x[5] - pi)])
            self.robot_config.save_file()

        if len(data[self.odom_list[0]]) >= self.max_opt_keyframes -1:
            print( 'Calibration finished.' )
            self.signal_shutdown()

        return result


    def create_data( self ):
        """ creates the data that we use for the optimization process. we collect the delta movement between keyframes and return it.
        the returned object is a dictionary with the odom topic names as keys. each topic has a list attached, that contains (dx,dy,dtheta)
        tuples, effectifly a 2d array, though not a numpy one. """
        data = dict()
        for key in self.keyframes:  # iterate over every odom topic
            odom = self.keyframes[key]      # odom keyframes for a single topic
            length = len(odom)              # number of keyframes (same number for each odom topic)
            deltas = []                     # list where we store the odometry deltas
            for i in range(1,length):       # create all odometry deltas for a single topic
                old_frame = TF2d.from_xyt( odom[i-1][0], odom[i-1][1], odom[i-1][2] )
                new_frame = TF2d.from_xyt( odom[i][0], odom[i][1], odom[i][2] )
                delta_tf = new_frame * old_frame.inverse()
                deltas.append( DataPoint.from_xyt(delta_tf.as_xyt()) )
                #dx = odom[i][0] - odom[i-1][0]
                #dy = odom[i][1] - odom[i-1][1]
                #dt = fix_rad_range( odom[i][2] - odom[i-1][2] )
                #deltas.append( (dx,dy,dt) )
            data[key] = deltas
        #self.debug_data( data )
        return data
    
    def debug_data( self, data ):
        """ prints debug messages """
        ref_topic = self.odom_list[0]
        print( 'keyframes.keys=', self.keyframes.keys() )
        print( 'data.keys=', data.keys() )
        length = len( data[ref_topic] )
        for i in range(length):
            a = self.odom_list[0]
            b = self.odom_list[1]
            print( 'dt={},   {}'.format(data[a][i][2], data[b][i][2]) )
    
    def save_data_async( self, data=None ):
        if not data:
            data = self.create_data()
        filename = self.save_data
        new_thread = Thread( target=self.save_data_points, args=(data, filename) )
        new_thread.start()
    
    def save_data_points( self, data, filename ):
        """ saves the data points to the provided file"""
        if not data:
            data = self.create_data()
        with open(filename, 'w') as file_handle:
            pickle.dump( data, file_handle, pickle.HIGHEST_PROTOCOL )
            
    def load_data_points( sel, filename ):
        """ loads data points from the provided file"""
        with open(filename, 'w') as file_handle:
            data = pickle.load( file_handle )
        return data

    def signal_shutdown(self):
        self.republisher.front_sub.unregister()
        self.republisher.back_sub.unregister()
        rospy.signal_shutdown("Finished!")



if __name__ == '__main__':
    node = CalibratorNode()
    rospy.spin()


