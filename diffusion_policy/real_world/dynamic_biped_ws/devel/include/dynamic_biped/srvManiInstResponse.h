// Generated by gencpp from file dynamic_biped/srvManiInstResponse.msg
// DO NOT EDIT!


#ifndef DYNAMIC_BIPED_MESSAGE_SRVMANIINSTRESPONSE_H
#define DYNAMIC_BIPED_MESSAGE_SRVMANIINSTRESPONSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace dynamic_biped
{
template <class ContainerAllocator>
struct srvManiInstResponse_
{
  typedef srvManiInstResponse_<ContainerAllocator> Type;

  srvManiInstResponse_()
    : stateRes(0)  {
    }
  srvManiInstResponse_(const ContainerAllocator& _alloc)
    : stateRes(0)  {
  (void)_alloc;
    }



   typedef int8_t _stateRes_type;
  _stateRes_type stateRes;





  typedef boost::shared_ptr< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> const> ConstPtr;

}; // struct srvManiInstResponse_

typedef ::dynamic_biped::srvManiInstResponse_<std::allocator<void> > srvManiInstResponse;

typedef boost::shared_ptr< ::dynamic_biped::srvManiInstResponse > srvManiInstResponsePtr;
typedef boost::shared_ptr< ::dynamic_biped::srvManiInstResponse const> srvManiInstResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator1> & lhs, const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator2> & rhs)
{
  return lhs.stateRes == rhs.stateRes;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator1> & lhs, const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dynamic_biped

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "556abd838730602dd98183f793a0e26b";
  }

  static const char* value(const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x556abd838730602dULL;
  static const uint64_t static_value2 = 0xd98183f793a0e26bULL;
};

template<class ContainerAllocator>
struct DataType< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dynamic_biped/srvManiInstResponse";
  }

  static const char* value(const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int8 stateRes\n"
"\n"
;
  }

  static const char* value(const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.stateRes);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct srvManiInstResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dynamic_biped::srvManiInstResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dynamic_biped::srvManiInstResponse_<ContainerAllocator>& v)
  {
    s << indent << "stateRes: ";
    Printer<int8_t>::stream(s, indent + "  ", v.stateRes);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DYNAMIC_BIPED_MESSAGE_SRVMANIINSTRESPONSE_H
