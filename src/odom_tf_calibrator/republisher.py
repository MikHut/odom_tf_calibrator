#!/usr/bin/env python

from __future__ import print_function


import rospy
import tf
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler



class Republisher( object ):
    """ re-publishes the /scan topics with the calibration applied to them. we currently assume a thorvald robot with one front laser and one back laser """
    def __init__( self, scan_topics ):
        self.topic_prefix = 'calib'
        self.scan_topics = scan_topics
        self.params = []
        self.counter = 0
        for t in self.scan_topics:
            self.params.append( 0.0 )
            self.params.append( 0.0 )
            self.params.append( 0.0 )
        self.front_sub = rospy.Subscriber(self.scan_topics[0], LaserScan, self.callback_front )
        self.back_sub = rospy.Subscriber(self.scan_topics[1], LaserScan, self.callback_back )
        self.pub_front = rospy.Publisher( '/' + self.topic_prefix + self.scan_topics[0], LaserScan, queue_size=2 )
        self.pub_back = rospy.Publisher( '/' + self.topic_prefix + self.scan_topics[1], LaserScan, queue_size=2 )
        self.tfb = tf.TransformBroadcaster()
        print( 'republisher up and running' )
    
    def update_tf( self, params ):
        """ the method is called externaly (e.g. after an optimization run) to update the transformations.
           params is a 1d array containg the x,y,t transformation parameters for each sensor, e.g. for two sensors (x1,y1,t1, x2,y2,t2) """
        self.params = params
    
    def callback_front( self, msg ):
        new_frame = self.topic_prefix + '/' + msg.header.frame_id
        msg.header.frame_id = new_frame
        self.pub_front.publish( msg )
        self.tfb.sendTransform( (self.params[0], self.params[1], 0.0), quaternion_from_euler(0, 0, self.params[2]), msg.header.stamp, new_frame, 'base_link' )
        #self.counter += 1
        #if self.counter % 60 == 0:
        #    print( (self.params[0], self.params[1], 0.0), quaternion_from_euler(0, 0, self.params[2]), rospy.Time.now(), new_frame, 'base_link' )

    def callback_back( self, msg ):
        new_frame = self.topic_prefix + '/' + msg.header.frame_id
        msg.header.frame_id = new_frame
        self.pub_back.publish( msg )
        self.tfb.sendTransform( (self.params[3], self.params[4], 0.0), quaternion_from_euler(0, 0, self.params[5]), msg.header.stamp, new_frame, 'base_link' )



if __name__ == '__main__':
    print( 'standalone operation not supported at the moment' )
    
    