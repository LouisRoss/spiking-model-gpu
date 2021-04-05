#include <iostream>
#include <memory>
#include <exception>

#include "libsocket/exception.hpp"
#include "libsocket/inetserverstream.hpp"
#include "libsocket/select.hpp"
#include "libsocket/socket.hpp"

#include <cuda_runtime_api.h>

#include "NeuronRecord.h"
#include "ModelRunner.h"
#include "GpuModelCarrier.h"
#include "GpuModelHelper.h"
#include "GpuModelUi.h"
#include "ICommandControlAcceptor.h"
#include "CommandControlListenSocket.h"
#include "QueryResponseListenSocket.h"
#include "CommandControlHandler.h"

using std::cout;
using std::unique_ptr;
using std::make_unique;

using namespace embeddedpenguins::gpu::neuron::model;
using embeddedpenguins::core::neuron::model::KeyListener;
using embeddedpenguins::core::neuron::model::ICommandControlAcceptor;
using embeddedpenguins::core::neuron::model::CommandControlListenSocket;
using embeddedpenguins::core::neuron::model::QueryResponseListenSocket;
using embeddedpenguins::core::neuron::model::CommandControlHandler;
using embeddedpenguins::gpu::neuron::model::GpuModelCarrier;
using embeddedpenguins::gpu::neuron::model::GpuModelHelper;
using embeddedpenguins::gpu::neuron::model::GpuModelUi;

int RunServer();


///////////////////////////////////////////////////////////////////////////
//Main program entry.
//Run the brain map.
//
int main(int argc, char* argv[])
{
	if (cuda::device::count() == 0) {
		cout << "No CUDA devices on this system\n";
        return -1;
	}

    ModelRunner<NeuronRecord> modelRunner(argc, argv);
    const auto& configuration = modelRunner.getConfigurationRepository();

    GpuModelCarrier carrier;
    GpuModelHelper<NeuronRecord> helper(carrier, configuration);
    if (!modelRunner.Run(carrier, helper))
    {
        cout << "Cannot run model, stopping\n";
        return 1;
    }

    //RunServer();
    try
    {
        //unique_ptr<ICommandControlAcceptor> commandControl = make_unique<CommandControlListenSocket>("localhost", "8000");

        //GpuModelUi ui(modelRunner, helper, make_unique<CommandControlListenSocket>("localhost", "8000"));
        GpuModelUi ui(
            modelRunner, 
            helper, 
            std::move(make_unique<QueryResponseListenSocket>(
                "localhost", 
                "8000",
                [](){
                    cout << "Callback lambda creating new CommandControlHandler\n";
                    return std::move(make_unique<CommandControlHandler>());
                }
            ))
        );
        ui.ParseArguments(argc, argv);
        ui.PrintAndListenForQuit();
    } catch (libsocket::socket_exception ex)
    {
        cout << "Caught exception " << ex.mesg << "\n";
    }
    catch (std::exception ex)
    {
        cout << "Caught exception " << ex.what() << "\n";
    }

    modelRunner.WaitForQuit();
    return 0;
}

int RunServer()
{
    using std::string;
    using std::unique_ptr;

    using libsocket::inet_stream;
    using libsocket::inet_stream_server;
    using libsocket::selectset;

    //string host = "::1";
    string host = "localhost";
    string port = "8000";
    string answ;

    try {
        inet_stream_server srv(host, port, LIBSOCKET_IPv4);

        selectset<inet_stream_server> set1;
        set1.add_fd(srv, LIBSOCKET_READ);

        for (;;) {
            /********* SELECT PART **********/
            std::cout << "Called select()\n";

            libsocket::selectset<inet_stream_server>::ready_socks
                readypair;  // Create pair (libsocket::fd_struct is the return
                            // type of selectset::wait()

            readypair = set1.wait();  // Wait for a connection and save the pair
                                      // to the var

            inet_stream_server* ready_srv = dynamic_cast<inet_stream_server*>(
                readypair.first
                    .back());  // Get the last fd of the LIBSOCKET_READ vector
                               // (.first) of the pair and cast the socket* to
                               // inet_stream_server*

            readypair.first.pop_back();  // delete the fd from the pair

            std::cout << "Ready for accepting\n";

            /*******************************/

            unique_ptr<inet_stream> cl1 = ready_srv->accept2();

            *cl1 << "Hello\n";

            answ.resize(100);

            *cl1 >> answ;

            std::cout << answ;

            // cl1 is closed automatically when leaving the scope!
        }

        srv.destroy();

    } catch (const libsocket::socket_exception& exc) {
        std::cerr << exc.mesg << std::endl;
    }
    return 0;
}
