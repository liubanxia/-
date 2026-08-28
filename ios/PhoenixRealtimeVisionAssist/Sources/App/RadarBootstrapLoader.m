#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <objc/message.h>

@interface LiteViewRadarAutoLoader : NSObject
@end

@implementation LiteViewRadarAutoLoader

+ (void)invokeSwiftBootstrap {
    dispatch_async(dispatch_get_main_queue(), ^{
        Class cls = NSClassFromString(@"LiteViewRadarBootstrap");
        SEL selector = NSSelectorFromString(@"install");
        if (cls && [cls respondsToSelector:selector]) {
            ((void (*)(id, SEL))objc_msgSend)(cls, selector);
        }
    });
}

+ (void)scheduleBootstrapAttempts {
    // Swift/Objective-C runtime registration and the first UIWindowScene do not always
    // become ready in the same launch phase. Re-issuing this idempotent install call
    // is cheap, and LiteViewRadarBootstrap itself refuses duplicate launcher views.
    static const NSTimeInterval delays[] = {0.0, 0.15, 0.5, 1.0, 2.0, 4.0};
    const NSUInteger count = sizeof(delays) / sizeof(delays[0]);

    for (NSUInteger index = 0; index < count; index++) {
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delays[index] * NSEC_PER_SEC)),
            dispatch_get_main_queue(),
            ^{
                [self invokeSwiftBootstrap];
            }
        );
    }
}

+ (void)load {
    // Do not rely on one early +load lookup. On SwiftUI launches the Swift class and
    // key UIWindowScene may become available later than this Objective-C object.
    [self scheduleBootstrapAttempts];

    NSNotificationCenter *center = NSNotificationCenter.defaultCenter;
    [center addObserverForName:UIApplicationDidFinishLaunchingNotification
                        object:nil
                         queue:NSOperationQueue.mainQueue
                    usingBlock:^(__unused NSNotification *note) {
        [self scheduleBootstrapAttempts];
    }];

    [center addObserverForName:UISceneDidActivateNotification
                        object:nil
                         queue:NSOperationQueue.mainQueue
                    usingBlock:^(__unused NSNotification *note) {
        [self scheduleBootstrapAttempts];
    }];
}

@end
