from twisted.internet.ssl import CertificateOptions
options = CertificateOptions()
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from twisted.internet.defer import inlineCallbacks

app_runner = ApplicationRunner(u"wss://api.poloniex.com", u"realm1", ssl=options)

class MyTicker(ApplicationSession):

    @inlineCallbacks
    def onJoin(self, details):

        # 1. subscribe to a topic so we receive events
        def onevent(msg):
            print("Got event: {}".format(msg))

        yield self.subscribe(onevent, 'ticker')

import ipdb; ipdb.set_trace()
app_runner.run(MyTicker)