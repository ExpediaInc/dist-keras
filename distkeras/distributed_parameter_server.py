import copy

import math

import numpy as np

import socket

import threading

from distkeras.networking import recv_data
from distkeras.networking import send_data
from distkeras.utils import deserialize_keras_model
from distkeras.networking import connect

class DistributedParameterServer(object):
    """Abstract class which provides basic attributed and methods for all
       parameter servers.

    # Arguments
        model: string. Serialized Keras model.
               See: distkeras.utils.serialize_keras_model
    """

    def __init__(self, model):
        self.model = deserialize_keras_model(model)
        self.num_updates = 0

    def initialize(self):
        """Initializes the parameter server.

        This method is called after self.start().
        """
        raise NotImplementedError

    def start(self):
        """Starts the parameter server in a new thread."""
        raise NotImplementedError

    def run(self):
        """Main event loop of the parameter server."""
        raise NotImplementedError

    def stop(self):
        """Notifies the parameter server thread to stop."""
        raise NotImplementedError

    def get_model(self):
        """Returns the Keras model which will be trained by the workers."""
        return self.model

    def reset_update_counter(self):
        """Resets the model update counter."""
        self.num_updates = 0

    def get_num_updates(self):
        """Returns the number of model updates the parameter server has performed."""
        return self.num_updates


class SocketDistributedParameterServer(DistributedParameterServer):
    """Abstract class of a parameter server which is based on a socket implementation.

    This means that this parameter server accepts multiple TCP connections from multiple
    workers, and uses a costum protocol to transmit and receive the model parameters. This
    is done by implementing a custom protocol. Which is fully described in the
    distkeras.networking module.

    # Arguments
        model: string. Serialized Keras model.
               See: distkeras.utils.serialize_keras_model
        port: int. Listing port number.
    """

    def __init__(self, model, ip_list, port=5000, num_children=3,com_window =10):
        super(SocketDistributedParameterServer, self).__init__(model)
        self.master_port = port
        self.socket_parent = None
        self.socket_child = None
        self.running = False
        self.connections = []
        self.mutex = threading.Lock()
        self.parent_ip = self.find_parent_ip(ip_list, num_children) 
        self.disable_nagle = True
        self.com_window = com_window
        self.finished_children_count = 0
        self.connected_children_and_excutor_count = 0

    def find_parent_ip(self,ip_list,num_children):
        host_ip = socket.gethostbyname(socket.gethostname())
        for i in range(len(ip_list)):
            if ip_list[i] == host_ip:
                if i >0:
                    return ip_list[int( (i-1) / num_children)]
                else:
                    return ip_list[0]

    def next_executor_update(self):
        """Increments the number of model updates by 1./com_window"""
        self.num_updates += 1.0 / self.com_window

    def next_child_update(self):
        """Increments the number of model updates by 1."""
        self.num_updates += 1.0

    def initialize(self):
        """Sets up the listing port."""
        # Reset the running flag.
        self.running = True
        # Prepare a socket.
        file_descriptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm.
        file_descriptor.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Check if the master port needs to be assigned by the OS.
        if self.master_port is None:
            file_descriptor.bind(('0.0.0.0', 0))
            # Retrieve the port assigned by the OS.
            self.master_port = int(file_descriptor.getsockname()[1])
        else:
            file_descriptor.bind(('0.0.0.0', self.master_port))
        # Listen to the socket.
        file_descriptor.listen(5)
        # Assign the socket.
        self.socket_child = file_descriptor

    def handle_executor_commit(self, conn, addr):
        """Handles parameter updates coming from the executor workers.

        # Arguments:
            conn: socket. The opened connection.
            addr: addr. Address of the remote host.
        """
        raise NotImplementedError

    def handle_child_commit(self, conn, addr):
        """Handles parameter updates coming from the children workers.

        # Arguments:
            conn: socket. The opened connection.
            addr: addr. Address of the remote host.
        """
        raise NotImplementedError

    def handle_pull(self, conn, addr):
        """Handles parameter requests coming from the workers. This will
        actually send the model parameters to the requesting host.

        # Arguments:
            conn: socket. The opened connection.
            addr: addr. Address of the remote host.
        """
        # Fetch the raw center variables.
        with self.mutex:
            cv = copy.deepcopy(self.center_variable)
        # Send the data over the socket.
        send_data(conn, cv)

    def connect(self):
        """Connect with the remote parameter server."""
        self.socket_parent = connect(self.parent_ip, self.master_port, self.disable_nagle)

    def pull(self):
    	#establish the connection
    	if self.socket_parent is None:
    		self.connect()
        """Requests the center variable from the parameter server."""
        # Request a pull from the parameter server.
        self.socket_parent.sendall(b'p')
        # Fetch the center variable from the parent parameter server.
        temp = np.asarray(recv_data(self.socket_parent))
        with self.mutex:
            #add the culmulated commit from children
            self.center_variable = temp + self.center_variable - self.center_variable_old
        self.center_variable_old = temp
        #unblock the commit
        self.block_commit_to_parent = False

    def commit(self, residual):
        """Sends the gradient residual to the parameter server."""
        raise NotImplementedError


    def cancel_accept(self):
        """This method will cancel the accept procedure. The method
        is meant to be executed by the stop() procedure.
        """
        file_descriptor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to the listening socket to cancel the accept.
            file_descriptor.connect(("localhost", self.master_port))
            file_descriptor.close()
        except Exception as e:
            print(e)

    def commit_and_pull_from_parent(self):
        """This method will try to commit to and pull from parent parameter server.
        Only one thread can do commit and pull in the same time
        """
        if not self.block_commit_to_parent:
            ready_to_commit_bool = False
            with self.mutex:
                if self.get_num_updates() > 1.0 :
                    #reset the counter
                    self.reset_update_counter()
                    ready_to_commit_bool = True
                    residual = self.center_variable - self.center_variable_old
                    self.block_commit_to_parent =True
                    self.center_variable_old = copy.deepcopy(self.center_variable)
            if ready_to_commit_bool:
                self.commit(residual) #no need to hold mutex
                self.pull() 

    def handle_connection(self, conn, addr):
        """
        A parameter server has two main functionalities. Nodes are able to
        pull (p) the current state, or 'commit' a state. This is implemented
        in the following functionality. Classes which implement these interfaces
        should not worry about connection handling.
        """
        try:
            while self.running:
                # Fetch the current action.
                action = conn.recv(1).decode()
                # Check if the action is a commit (most of the cases).
                if action == 'c':
                    # Handle the commit.
                    self.handle_executor_commit(conn, addr)
                if action == 'h':
                    # Handle the commit.
                    self.handle_child_commit(conn, addr)
                elif action == 'p':
                    # Handle the pull.
                    self.handle_pull(conn, addr)
                elif action =='s':
                    self.finished_children_count += 1
                    print("finished_children_count = " +str(self.finished_children_count))
                #check if need commit the weight to parent server
                #only one thread is able to commit to and pull from parent server
                self.commit_and_pull_from_parent()
        except Exception as e:
            print(e)

    def start(self):
        """Starts the parameter server."""
        # Set the running flag.
        self.running = True
        self.block_commit_to_parent = False

    def run(self):
        print("""Main event loop of the parameter server.""")
        # Listen for incoming connections.
        while self.running:
            try:
                # Accept incoming connections.
                conn, addr = self.socket_child.accept()
                # Handle the connection.
                thread = threading.Thread(target=self.handle_connection, args=(conn, addr))
                thread.start()
                # Store the connection in the dictionary.
                self.connections.append(thread)
                self.connected_children_and_excutor_count += 1
                print("connected_children_and_excutor_count = " +str(self.connected_children_and_excutor_count))
            except Exception as e:
                print(e)
        print("stopped parameter server running")

    def stop(self):
        """Stop the parameter server. This will also cleanup all existing connections."""
        self.running = False
        # Check if a socket is allocated.
        if self.socket_child:
            #need to figure out why the threads cannot join successfully
            #self.cleanup_connections()
            self.finalize()
            self.socket_child.close()
            self.cancel_accept()
            self.socket_child = None

        self.connections = []
        """notify its parents to stop"""
        self.socket_parent.sendall(b's')

    def finalize(self):
        """Method that is called when the parameter server stops."""
        print("Not executed")

    def cleanup_connections(self):
        """Clean all existing connections up."""
        # Iterate over all connections.
        for thread in self.connections:
            # Fetch the thread object.
            thread.join()
            del thread

class ADAGDistributedParameterServer(SocketDistributedParameterServer):
    """A parameter server which integrates the incoming gradient residuals into
       the model, and integrates them using the ADAG scheme.

    # Arguments
        model: string. Keras model.
               See: distkeras.utils.serialize_keras_model
        master_port: int. Port number of the parameter server.
    """

    def __init__(self, model, master_port,ip_list, num_children=3,com_window =10):
        super(ADAGDistributedParameterServer, self).__init__(model, ip_list, master_port, num_children,com_window)
        self.center_variable = np.asarray(self.model.get_weights())
        self.center_variable_old = np.asarray(self.model.get_weights())
        self.com_window = com_window
    def handle_executor_commit(self, conn, addr):
        # Receive the parameters from the remote node.
        data = recv_data(conn)
        # Extract the data from the dictionary.
        r = data['residual']
        with self.mutex:
            # Update the center variable.
            self.center_variable = self.center_variable + 1.0/self.com_window * r
            # Increment the number of parameter server updates.
            self.next_executor_update()

    def handle_child_commit(self, conn, addr):
        # Receive the parameters from the remote node.
        data = recv_data(conn)
        # Extract the data from the dictionary.
        r = data['residual']
        with self.mutex:
            # Update the center variable.
            self.center_variable = self.center_variable + r
            # Increment the number of parameter server updates.
            self.next_child_update()

    def handle_pull(self, conn, addr):
        """Handles parameter requests coming from the workers. This will
        actually send the model parameters to the requesting host.

        # Arguments:
            conn: socket. The opened connection.
            addr: addr. Address of the remote host.
        """
        # Fetch the raw center variables.
        with self.mutex:
            cv = copy.deepcopy(self.center_variable)
            # Send the data over the socket.
        send_data(conn, cv)

    def finalize(self):
        # Set the weights of the model.
        self.model.set_weights(self.center_variable)

    def commit(self, residual):
        #establish the connection to its parent
        #need a way to coordinate the parent socket connection setup between different machine
        if self.socket_parent is None:
            self.connect()
        data = {}
        data['worker_id'] = -1
        data['residual'] = residual
        # Request a commit from the parameter server.
        self.socket_parent.sendall(b'h')
        # Send the data to the paramter server.
        send_data(self.socket_parent, data)