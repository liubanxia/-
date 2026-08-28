#import <Foundation/Foundation.h>
#import <objc/message.h>

@interface LiteViewRadarAutoLoader : NSObject
@end

@implementation LiteViewRadarAutoLoader

+ (void)load {
    dispatch_async(dispatch_get_main_queue(), ^{
        Class cls = NSClassFromString(@"LiteViewRadarBootstrap");
        SEL selector = NSSelectorFromString(@"install");
        if (cls && [cls respondsToSelector:selector]) {
            ((void (*)(id, SEL))objc_msgSend)(cls, selector);
        }
    });
}

@end
